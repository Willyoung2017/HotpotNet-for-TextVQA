# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np
import torch
from mmf.common.sample import Sample
from mmf.datasets.builders.textvqa.dataset import TextVQADataset
from mmf.datasets.mmf_dataset import MMFDataset
from mmf.utils.distributed import byte_tensor_to_object, object_to_byte_tensor
from mmf.utils.text import word_tokenize
import random


class TextVQAHotpotDataset(TextVQADataset):
    def __init__(self, config, dataset_type, imdb_file_index, *args, **kwargs):
        super().__init__(
            config, dataset_type, imdb_file_index, *args, **kwargs
        )
        self._name = "textvqa_hotpot"
        object_clsname_path = self.config.object_clsname_path
        self.object_clsname = [x.strip() for x in list(open(object_clsname_path, 'r'))]
        self.object_clsname = ['background'] + self.object_clsname
        ## set mode by checking feature path (rosetta/msocr)
        self.msocr = 'ocr_en_frcn' not in config.features.train[0]
        if self.msocr: assert ('fc6' not in config.features.train[0])
        self.do_object_filter = config.get('do_object_filter', True)
        if not self.do_object_filter:
            print('*'*20+'WARNING: object filter is disabled'+'*'*20)

    def preprocess_sample_info(self, sample_info):
        return sample_info  # Do nothing

    def postprocess_evalai_entry(self, entry):
        return entry  # Do nothing

    def format_for_prediction(self, report):
        answer_processor = self.answer_processor

        batch_size = len(report.question_id)
        pred_answers = report.scores.argmax(dim=-1).view(batch_size, -1)
        answer_space_size = answer_processor.get_true_vocab_size()

        image_ids = report.image_id.cpu().numpy()
        context_tokens = report.context_tokens.cpu().numpy()
        predictions = []
        for idx, question_id in enumerate(report.question_id):
            # collect VQA answers
            image_id = byte_tensor_to_object(image_ids[idx])
            tokens = byte_tensor_to_object(context_tokens[idx])
            answer_words = []
            pred_source = []
            for answer_id in pred_answers[idx].tolist():
                if answer_id >= answer_space_size:
                    answer_id -= answer_space_size
                    answer_words.append(word_tokenize(tokens[answer_id]))
                    pred_source.append("OCR")
                else:
                    if answer_id == answer_processor.EOS_IDX:
                        break
                    answer_words.append(
                        answer_processor.answer_vocab.idx2word(answer_id)
                    )
                    pred_source.append("VOCAB")
            # join all the answer tokens with space
            # (this should be correct for almost all cases)
            pred_answer = " ".join(answer_words).replace(" 's", "'s")
            entry = {
                "question_id": question_id.item(),
                "image_id": image_id,
                "answer": pred_answer,
                "pred_source": pred_source,
            }
            entry = self.postprocess_evalai_entry(entry)

            predictions.append(entry)

        return predictions

    def __getitem__(self, idx):
        sample_info = self.annotation_db[idx]
        sample_info = self.preprocess_sample_info(sample_info)
        current_sample = Sample()

        # breaking change from VQA2Dataset: load question_id
        current_sample.question_id = torch.tensor(
            sample_info["question_id"], dtype=torch.int
        )

        if isinstance(sample_info["image_id"], int):
            current_sample.image_id = str(sample_info["image_id"])
        else:
            current_sample.image_id = sample_info["image_id"]

        if self._use_features is True:
            features = self.features_db[idx]
            if 'object_tokens' not in features['image_info_0']:
                features['image_info_0']['object_tokens'] = \
                    [self.object_clsname[x] for x in features['image_info_0']['objects']]
            current_sample.update(features)
        current_sample = self.add_sample_details(sample_info, current_sample)
        current_sample = self.add_answer_info(sample_info, current_sample)

        # only the 'max_features' key is needed
        # pop other keys to minimize data loading overhead
        for k in list(current_sample.image_info_0):
            if k == 'conf' or k == 'num_boxes':
                current_sample.image_info_0.pop(k)
        return current_sample

    def add_sample_details(self, sample_info, sample):
        # Todo: add bert features
        # Todo: deduplicate object detection bbox
        sample.image_id = object_to_byte_tensor(sample.image_id)
        ocr_str_len, obj_str_len = 100, 100
        # ocr_str_len, obj_str_len = 100, 100   ## oom on 16g

        #######################################################################
        # 1. Load text (question words)
        # breaking change from VQA2Dataset:
        # load the entire question string, not tokenized questions, since we
        # switch to BERT tokenizer in M4C and do online tokenization
        question_str = (
            sample_info['question'] if 'question' in sample_info
            else sample_info['question_str']
        )

        processed_question = self.text_processor({"text": question_str}, updatelen=20)

        sample.text_mask_label = None
        if "input_ids" in processed_question:
            sample.text = processed_question["input_ids"]
            sample.text_len = torch.tensor(
                len(processed_question["tokens"]), dtype=torch.long
            )
        else:
            # For GLoVe based processors
            sample.text = processed_question["text"]
            sample.text_len = processed_question["length"]

        assert sample.text.shape[0] == 20, sample.text.shape
        #######################################################################
        # 2. Load object
        ## Load object bert token
        object_tokens = sample['image_info_0']['object_tokens']
        # Get FastText embeddings for object label tokens
        object_context = self.context_processor({"tokens": object_tokens})
        if self.do_object_filter:
            indices = [i for i, x in enumerate(object_context["tokens"]) if x != "background"]
        else:
            indices = [i for i in range(len(object_context["tokens"]))]
        object_tokens = [object_tokens[i] for i in indices]
        object_context = self.context_processor({"tokens": object_tokens})
        # object bounding box information
        obj_bbox = sample['image_info_0']['bbox'] * [1. / sample_info["image_width"], 1. / sample_info["image_height"],
                                                     1. / sample_info["image_width"], 1. / sample_info["image_height"]]
        obj_bbox = [obj_bbox[i] for i in indices]
        if len(obj_bbox) == 0:
            obj_bbox = np.zeros(
                (0, 4), np.float32
            )
        obj_max_len = self.config.processors.copy_processor.params.obj_max_length
        sample.obj_bbox_coordinates = self.copy_processor(
            {"blob": np.array(obj_bbox, dtype=np.float32)}
        )["blob"][:obj_max_len]
        # if "background" in sample['image_info_0']['object_tokens']:
        #     with open("test.pkl", "wb") as f:
        #         import pickle
        #         pickle.dump((question_str, sample["image_info_0"]["bbox"], sample['image_info_0']['object_tokens'], sample_info["image_path"]), f)
        #     print("background token found")

        # obj_str = ' '.join(object_tokens)
        # processed_obj = self.text_processor({"text": obj_str}, updatelen=obj_str_len)
        # sample.objtext_mask_label = None
        # sample.obj_text = processed_obj["input_ids"]
        # sample.obj_text_len = torch.tensor(
        #     len(processed_obj["tokens"]), dtype=torch.long
        # )
        sample.object_context = object_context["text"]
        sample.object_tokens = object_context["tokens"]
        sample.object_tokens = object_to_byte_tensor(sample.object_tokens)
        sample.object_feature_0 = object_context["text"]
        sample.object_info_0 = Sample()
        sample.object_info_0.max_features = object_context["text"].shape[0]
        # sample.object_overlap = self.region_overlap(sample.obj_bbox_coordinates, sample.obj_bbox_coordinates,
        #                                      threshold=0.9)
        #######################################################################
        # 3. Load OCR
        if not self.use_ocr:
            # remove all OCRs from the sample
            # (i.e. make an empty OCR list)
            sample_info['ocr_tokens'] = []
            sample_info['ocr_info'] = []
            if 'ocr_normalized_boxes' in sample_info:
                sample_info['ocr_normalized_boxes'] = np.zeros(
                    (0, 4), np.float32
                )
            # clear OCR visual features
            sample.image_feature_1 = torch.zeros_like(sample.image_feature_1)
        # Preprocess OCR tokens
        ## ocr_token_processor for lower(), etc.
        if self.msocr:  ## load ms ocr prediction directly from _info.npy; else: from imdb provided
            max_len = self.config.processors.answer_processor.params.max_length
            if 'ocr_conf' not in sample.image_info_1: sample.image_info_1['ocr_conf'] = None
            if len(sample.image_info_1["ocr_tokens"]) > max_len:
                sample.image_info_1['ocr_tokens'], sample.image_info_1['ocr_boxes'] = \
                    self.ocr_truncate(np.array(sample.image_info_1['ocr_tokens']), sample.image_info_1['ocr_boxes'], \
                                      conf_array=np.array(sample.image_info_1['ocr_conf']), mode='naive')
            ocr_tokens = [
                self.ocr_token_processor({"text": token})["text"]
                for token in sample.image_info_1["ocr_tokens"]]
        else:
            print('not using msocr; duplicate warning; please manually comment this line')
            # Preprocess OCR tokens
            if hasattr(self, "ocr_token_processor"):
                ocr_tokens = [
                    self.ocr_token_processor({"text": token})["text"]
                    for token in sample_info["ocr_tokens"]
                ]
            else:
                ocr_tokens = sample_info["ocr_tokens"]

        ## Load OCR bert token
        ## might lead to "Token indices sequence length is longer than the specified maximum sequence length for this model" if too long
        ## but truncated to updatelen anyway
        # ocr_str = ' '.join(ocr_tokens)
        # processed_ocr = self.text_processor({"text": ocr_str}, updatelen=ocr_str_len)
        # sample.ocrtext_mask_label = None
        # sample.ocr_text = processed_ocr["input_ids"]
        # sample.ocr_text_len = torch.tensor(
        #     len(processed_ocr["tokens"]), dtype=torch.long
        # )
        #######################################################################
        # Get FastText embeddings for OCR tokens
        context = self.context_processor({"tokens": ocr_tokens})
        sample.context = context["text"]
        sample.ocr_tokens = context["tokens"]
        sample.context_tokens = object_to_byte_tensor(context["tokens"])
        sample.context_feature_0 = context["text"]
        sample.context_info_0 = Sample()
        sample.context_info_0.max_features = context["length"]

        # Get PHOC embeddings for OCR tokens
        if hasattr(self, "phoc_processor"):
            context_phoc = self.phoc_processor({"tokens": ocr_tokens})
            sample.context_feature_1 = context_phoc["text"]
            sample.context_info_1 = Sample()
            sample.context_info_1.max_features = context_phoc["length"]

        # OCR order vectors
        if self.config.get("use_order_vectors", False):
            order_vectors = np.eye(len(sample.ocr_tokens), dtype=np.float32)
            order_vectors = torch.from_numpy(order_vectors)
            order_vectors[context["length"] :] = 0
            sample.order_vectors = order_vectors
            # Get PHOC embeddings for OCR tokens

        if hasattr(self, "phoc_processor"):
            context_phoc = self.phoc_processor({"tokens": ocr_tokens})
            sample.context_feature_1 = context_phoc["text"]
            sample.context_info_1 = Sample()
            sample.context_info_1.max_features = context_phoc["length"]

        # OCR order vectors
        if self.config.get("use_order_vectors", False):
            order_vectors = np.eye(len(sample.ocr_tokens), dtype=np.float32)
            order_vectors = torch.from_numpy(order_vectors)
            order_vectors[context["length"] :] = 0
            sample.order_vectors = order_vectors

        # OCR bounding box information
        if self.msocr:
            max_len = self.config.processors.answer_processor.params.max_length
            ocr_bbox = np.zeros((0, 4), dtype=np.float32)
            if sample.image_info_1['ocr_boxes'].shape[0] != 0:
                ocr_bbox = sample.image_info_1['ocr_boxes'] * [1. / sample_info["image_width"],
                                                               1. / sample_info["image_height"],
                                                               1. / sample_info["image_width"],
                                                               1. / sample_info["image_height"]]
            sample.ocr_bbox_coordinates = self.copy_processor(
                {"blob": np.array(ocr_bbox, dtype=np.float32)}
            )["blob"][:max_len]
        else:
            if "ocr_normalized_boxes" in sample_info and hasattr(self, "copy_processor"):
                # New imdb format: OCR bounding boxes are already pre-computed
                max_len = self.config.processors.answer_processor.params.max_length
                sample.ocr_bbox_coordinates = self.copy_processor(
                    {"blob": sample_info["ocr_normalized_boxes"]}
                )["blob"][:max_len]
            elif self.use_ocr_info and "ocr_info" in sample_info:
                # Old imdb format: OCR bounding boxes are computed on-the-fly
                # from ocr_info
                sample.ocr_bbox_coordinates = self.bbox_processor(
                    {"info": sample_info["ocr_info"]}
                )["bbox"].coordinates

        #######################################################################
        ## RPP objective
        ## add OCR, OBJ region spatial location info
        ## If OCR region falls inside OBJ
        sample.overlap = self.region_overlap(sample.obj_bbox_coordinates, sample.ocr_bbox_coordinates[:, :4],
                                             threshold=0.99)
        targetoverlap = torch.tensor(0) if random.random() < 0.5 else torch.tensor(1)
        index = torch.where(sample.overlap == targetoverlap)
        len_index = int(index[0].shape[0])
        if len_index != 0:
            ind = random.randint(0, len_index - 1)
            sample.overlap = targetoverlap
            sample.overlap_obj, sample.overlap_ocr = index[0][ind], index[1][ind]
        else:
            sample.overlap = torch.tensor(-1)
            sample.overlap_obj, sample.overlap_ocr = torch.tensor(0), torch.tensor(0)
        return sample

    def add_answer_info(self, sample_info, sample):
        # Load real answers from sample_info
        answers = sample_info.get("answers", [])
        answer_processor_arg = {"answers": answers}

        answer_processor_arg["tokens"] = sample.pop("ocr_tokens", [])

        processed_answers = self.answer_processor(answer_processor_arg)

        assert not self.config.fast_read, (
            "In TextVQADataset, online OCR sampling is incompatible "
            "with fast_read, so fast_read is currently not supported."
        )

        sample.update(processed_answers)
        sample.answers = object_to_byte_tensor(answers)

        if "answers_scores" in sample:
            sample.targets = sample.pop("answers_scores")

        return sample

    def ocr_truncate(self, token_array, ocr_boxes, conf_array=None, mode='naive'):
        max_len = self.config.processors.answer_processor.params.max_length
        if conf_array is None: mode = 'naive'
        ## default original order
        if 'naive' in mode:
            idx = np.array(range(ocr_boxes.shape[0]))[:max_len]
        return token_array[idx].tolist(), ocr_boxes[idx, :]

    def region_overlap(self, bbox_obj, bbox_ocr, threshold=0.99, bgimgsize_thes=0.25, bgobj_mask=None, eps=1e-6):
        ## (x1, y1, x2, y2)
        bboxmask_obj = (bbox_obj.sum(1) != 0).long().sum()
        bboxmask_ocr = (bbox_ocr.sum(1) != 0).long().sum()
        area_obj = (bbox_obj[:, 2] - bbox_obj[:, 0]) * (bbox_obj[:, 3] - bbox_obj[:, 1])
        area_ocr = (bbox_ocr[:, 2] - bbox_ocr[:, 0]) * (bbox_ocr[:, 3] - bbox_ocr[:, 1])

        lt = torch.max(bbox_obj[:, None, :2], bbox_ocr[:, :2])  # [N,M,2]
        rb = torch.min(bbox_obj[:, None, 2:], bbox_ocr[:, 2:])  # [N,M,2]

        wh = (rb - lt).clamp(min=0)  # [N,M,2]
        inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

        cover = area_obj.unsqueeze(1).repeat(1, area_ocr.shape[0]) / (area_obj[:, None] + area_ocr - inter)
        cover = (cover > threshold).long()
        cover[bboxmask_obj:, :] = -1
        cover[:, bboxmask_ocr:] = -1
        if bgobj_mask is None:
            bgobj_mask = (area_obj > bgimgsize_thes)
        cover[bgobj_mask, :] = -1
        return cover

    """
    From VilBert, dataset/concept_cap_dataset
    """

    def random_word(self, tokens, tokenizer, mask_prob=0.15):
        """
        Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
        :param tokens: list of str, tokenized sentence.
        :param tokenizer: Tokenizer, object used for tokenization (we need it's vocab here)
        :return: (list of str, list of int), masked tokens and related labels for LM prediction
        """
        assert tokenizer['[MASK]'] == 103
        assert tokenizer['[UNK]'] == 100
        mask_prob = mask_prob

        output_label = []
        token_inds = tokens['token_inds']
        for i, token in enumerate(token_inds):
            prob = random.random()
            # mask token with 15% probability

            if prob < mask_prob and token not in [0, 101, 102]:
                # append current token to output (we will predict these later)
                try:
                    # output_label.append(tokenizer.vocab[token])
                    output_label.append(int(token))
                except KeyError:
                    # For unknown words (should not occur with BPE vocab)
                    # output_label.append(tokenizer.vocab["[UNK]"])
                    output_label.append(100)
                    # logger.warning(
                    #     "Cannot find token '{}' in vocab. Using [UNK] insetad".format(token)
                    # )

                prob /= mask_prob

                # 80% randomly change token to mask token
                if prob < 0.8:
                    # token_inds[i] = "[MASK]"
                    token_inds[i] = 103

                # 10% randomly change token to random token
                elif prob < 0.9:
                    # token_inds[i] = random.choice(list(tokenizer.vocab.items()))[0]
                    token_inds[i] = random.choice(list(range(1000, len(tokenizer))))  # [0]

                # -> rest 10% randomly keep current token

            else:
                # no masking token (will be ignored by loss function later)
                output_label.append(-1)
        tokens['token_inds'] = token_inds
        return tokens, torch.tensor(output_label)

    def random_word_answer(self, processed_answers, mask_prob=0.15):
        pretrain_targets = processed_answers["answers_scores"] * 0
        pretrain_prev_inds = processed_answers["train_prev_inds"]
        pretrain_loss_mask = processed_answers["train_loss_mask"] * 0

        for ii in range(pretrain_prev_inds.shape[0]):
            prob = random.random()
            if prob < mask_prob and pretrain_prev_inds[ii] != 0 and pretrain_prev_inds[ii] != 1:
                pretrain_loss_mask[ii] = 1
                pretrain_targets[ii, :] = processed_answers["answers_scores"][ii - 1, :]
                prob /= mask_prob
                if prob < 0.8:
                    pretrain_prev_inds[ii] = 3  ## <unk>
                elif prob < 0.9:
                    pretrain_prev_inds[ii] = random.choice(list(range(4, pretrain_targets.shape[1] - 1)))
                else:
                    pretrain_loss_mask[ii] = 0
        processed_answers["answers_scores"] = pretrain_targets
        processed_answers["train_prev_inds"] = pretrain_prev_inds
        processed_answers["train_loss_mask"] = pretrain_loss_mask
        return
