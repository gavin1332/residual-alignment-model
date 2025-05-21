from pipeline import Pipeline

class Strategy(Pipeline):
    def init_properties(self):
        self.loss = 'ours_sft'
        self.args.dataset = 'uc'
        self.train_dataset = self.args.train_dataset or self.pairwise_dataset()
        self.warmup_steps = self.calc_examples_per_epoch() // self.args.batch_size // 10
        self.extra_training_args = {'loss.alpha': self.args.alpha}

    def add_args(self, arg_group):
        arg_group['train'].add_argument('--alpha', type=float, default=1e-4)
        arg_group['train'].add_argument('--train_dataset', type=str)

    def exp_name_postfix(self):
        alpha = f'_a{self.args.alpha}' if self.args.alpha != 1e-4 else ''
        return f'{alpha}{super().exp_name_postfix()}'

    def validate_args(self, parser):
        if not self.args.skip_sampling and not self.args.proposal:
            parser.error(f'--proposal is required in aligner sampling')


stategy = Strategy()
stategy.run()

