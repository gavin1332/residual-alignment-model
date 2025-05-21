from pipeline import Pipeline

class Strategy(Pipeline):
    def init_properties(self):
        self.loss = 'ours_sft'
        self.args.dataset = 'summ-sft'
        self.train_dataset = self.args.train_dataset or self.pairwise_dataset()
        self.warmup_steps = self.calc_examples_per_epoch() // self.args.batch_size // 10
        self.args.max_length = 1152
        self.args.max_prompt_length = 640
        self.eval_every = self.calc_examples_half_epoch()

    def add_args(self, arg_group):
        arg_group['train'].add_argument('--train_dataset', type=str)

    def validate_args(self, parser):
        if not self.args.skip_sampling and not self.args.proposal:
            parser.error(f'--proposal is required in aligner sampling')


stategy = Strategy()
stategy.run()

