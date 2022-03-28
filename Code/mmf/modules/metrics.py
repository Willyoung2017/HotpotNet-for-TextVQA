@registry.register_metric("lorra_textvqa_accuracy")
class LorraTextVQAAccuracy(TextVQAAccuracy):
    def calculate(self, sample_list, model_output, *args, **kwargs):
        model_output["scores"] = model_output["scores"][:, None, ...]
        return super().calculate(sample_list, model_output, *args, **kwargs)
