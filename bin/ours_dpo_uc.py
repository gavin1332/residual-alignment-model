from pipeline import Pipeline

class Strategy(Pipeline):
    def init_properties(self):
        self.loss = 'ours_dpo'
        self.args.dataset = 'uc'
        self.train_dataset = self.pairwise_dataset()
        self.warmup_steps = self.calc_examples_per_epoch() // self.args.batch_size // 10
        self.eval_every = self.calc_examples_per_epoch() // 2

    def validate_args(self, parser):
        if self.args.train and self.args.archive is None:
            parser.error(f'--archive is required in {self.loss} training')
        if not self.args.skip_sampling and not self.args.proposal:
            parser.error(f'--proposal is required in aligner sampling')


stategy = Strategy()
stategy.run()

