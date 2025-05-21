from pipeline import Pipeline
from typing import Dict
from fastchat.conversation import get_conv_template


class Strategy(Pipeline):
    def init_properties(self):
        self.loss = 'sft'
        self.args.dataset = 'hh'
        if self.args.warmup:
            self.train_dataset = 'hh_aligner_warmup'
            self.n_examples = (10000 // self.args.batch_size + 1) * self.args.batch_size
            self.warmup_steps = self.n_examples // self.args.batch_size // 2
        else:
            self.train_dataset = 'hh_aligner_common'
            self.warmup_steps = self.calc_examples_per_epoch() // self.args.batch_size // 10
        # extra tokens for "correction" indicator
        extra_length = 10
        self.args.max_length = 3072 + extra_length
        self.args.max_prompt_length = 2048 + extra_length
        self.use_vllm = True
        self.args.multiround = True
        self.args.for_aligner = True

    def add_args(self, arg_group):
        arg_group['train'].add_argument('--warmup', action='store_true')

    def exp_name(self):
        exp_name = super().exp_name()
        if self.args.warmup:
            exp_name = f'wu_{exp_name}'
        return exp_name

    def validate_args(self, parser):
        if self.args.train and not self.args.warmup and not self.args.archive:
            parser.error(f'--archive is required in aligner training')
        if (not self.args.skip_sampling or self.args.pre_sampling) and not self.args.proposal:
            parser.error(f'--proposal is required in aligner sampling')
        if not self.args.for_aligner:
            parser.error(f'--for_aligner is required in aligner training and sampling')
        if not self.args.multiround:
            parser.error(f'--multiround is required in aligner training and sampling')


stategy = Strategy()
stategy.run()

