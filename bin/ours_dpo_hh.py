from pipeline import Pipeline

class Strategy(Pipeline):
    def init_properties(self):
        self.loss = 'ours_dpo'
        self.args.dataset = 'hh'
        self.train_dataset = self.args.dataset
        self.warmup_steps = self.calc_examples_per_epoch() // self.args.batch_size // 10
        self.args.multiround = True
        self.eval_every = self.calc_examples_per_epoch() // 2
        self.extra_training_args = {'loss.beta': self.args.beta}

    def add_args(self, arg_group):
        arg_group['train'].add_argument('--beta', type=float, default=1.0)
        arg_group['train'].add_argument('--no_archive', action='store_true')

    def exp_name_postfix(self):
        beta = f'_b{self.args.beta}' if self.args.beta != 1.0 else ''
        return f'{beta}{super().exp_name_postfix()}'

    def validate_args(self, parser):
        if self.args.train and not self.args.archive and not self.args.no_archive:
            parser.error('--archive is required in ours_sft training')
        if not self.args.skip_sampling and not self.args.proposal:
            parser.error('--proposal is required in ours_sft sampling')


stategy = Strategy()
stategy.run()

