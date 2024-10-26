from FlagEmbedding.abc.evaluation import AbsEvalRunner

from .data_loader import MIRACLEvalDataLoader


class MIRACLEvalRunner(AbsEvalRunner):
    def load_data_loader(self) -> MIRACLEvalDataLoader:
        data_loader = MIRACLEvalDataLoader(
            eval_name=self.eval_args.eval_name,
            dataset_dir=self.eval_args.dataset_dir,
            cache_dir=self.eval_args.cache_path,
            token=self.eval_args.token
        )
        return data_loader
