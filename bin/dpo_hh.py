from pipeline import Pipeline

class Strategy(Pipeline):
    def init_properties(self):
        self.loss = 'dpo'
        self.args.dataset = 'hh'
        self.train_dataset = self.args.dataset
        self.warmup_steps = self.calc_examples_per_epoch() // self.args.batch_size // 10
        self.use_vllm = True
        self.args.multiround = True
        self.eval_every = self.calc_examples_per_epoch() // 2

    def validate_args(self, parser):
        if self.args.train and self.args.archive is None:
            parser.error(f'--archive is required in {self.loss} training')

stategy = Strategy()
stategy.run()

