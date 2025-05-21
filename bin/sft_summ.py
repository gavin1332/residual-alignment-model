from pipeline import Pipeline

class Strategy(Pipeline):
    def init_properties(self):
        self.loss = 'sft'
        self.args.dataset = 'summ-sft'
        self.train_dataset = self.args.dataset
        if self.args.warmup:
            self.n_examples = (10000 // self.args.batch_size + 1) * self.args.batch_size
            self.warmup_steps = self.n_examples // self.args.batch_size // 2
        else:
            self.warmup_steps = self.calc_examples_per_epoch() // self.args.batch_size // 10
        self.args.max_length = 1152 # 640 + 512
        self.args.max_prompt_length = 640
        self.use_vllm = True

    def add_args(self, arg_group):
        arg_group['train'].add_argument('--warmup', action='store_true')

    def exp_name(self):
        exp_name = super().exp_name()
        if self.args.warmup:
            exp_name = f'wu_{exp_name}'
        return exp_name


stategy = Strategy()
stategy.run()

