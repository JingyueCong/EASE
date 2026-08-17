import copy
import json
import torch
import datasets
from datasets import load_dataset

from .conv_util import create_template
from .datamodule import TrainDataModule, TorchDataset
from .ciru import factorial_dual_roles, load_ciru_units
from .f2r import load_f2r_pairs

class ToFU_DataModule(TrainDataModule):

    def __init__(
        self,
        split,
        tokenizer,
        conv_template_config,
        max_len=256,
        batch_size=8,
        with_retain=False,
        retain_num=400,
        with_dpo=False,
        expand_forget=False,
        with_perturb=False, # Our method
        r_sub_indices_path=None,  # Dual-ULD: path to JSON with R_sub indices
        data_role=None,           # Dual-ULD/F2R/F2D data-role identifier
        counterfactual_path=None,
        strict_retain_free=False,
        **kwargs,
    ):
        super().__init__()

        self.tokenizer = tokenizer
        self.max_len = max_len
        self.batch_size = batch_size
        self.dpo_mode = with_dpo
        self.conv_template = create_template(conv_template_config, tokenizer=tokenizer)

        def flatten_perturb(perturb_dataset):
            for sample in perturb_dataset:
                perturb_answer_list = sample.pop('perturbed_answer')
                newsample = copy.deepcopy(sample)
                for perturb_ans in perturb_answer_list[:1]:
                    newsample['answer'] = perturb_ans
                    yield newsample
        
        forget_eval = load_dataset('locuslab/TOFU', split)['train']
        cols_to_drop = [c for c in ['paraphrased_answer', 'paraphrased_question', 'perturbed_answer'] if c in forget_eval.column_names]
        if cols_to_drop:
            forget_eval = forget_eval.remove_columns(cols_to_drop)
        self.forget_eval = forget_eval

        # Strict F2R runs must not touch a retain split during training or
        # hyperparameter selection. Final utility evaluation is performed by
        # the separate open-unlearning evaluator after checkpoints are frozen.
        if strict_retain_free:
            self.retain_eval = None
        else:
            retain_eval = load_dataset('locuslab/TOFU', 'retain_perturbed')['train']
            retain_eval = retain_eval.remove_columns(['paraphrased_answer', 'paraphrased_question', 'perturbed_answer'])
            self.retain_eval = retain_eval

        perturb_eval = load_dataset('locuslab/TOFU', split)['train']
        if 'perturbed_answer' in perturb_eval.column_names:
            perturb_eval = datasets.Dataset.from_generator(flatten_perturb, gen_kwargs={"perturb_dataset": perturb_eval})
        self.perturb_eval = perturb_eval

        paraphrase_eval = load_dataset('locuslab/TOFU', split)['train']
        if 'paraphrased_answer' in paraphrase_eval.column_names:
            paraphrase_eval = paraphrase_eval.remove_columns(['answer', 'perturbed_answer', 'paraphrased_question'])
            paraphrase_eval = paraphrase_eval.rename_column('paraphrased_answer', 'answer')
        self.paraphrase_eval = paraphrase_eval

        # Construct training 
        base_forget_data = load_dataset('locuslab/TOFU', split)['train']
        base_retain_data = datasets.Dataset.from_dict({'question': [], 'answer': []})
        self.forget_length = len(base_forget_data)
        self.retain_length = 0
        if with_retain:
            if strict_retain_free:
                raise ValueError("strict_retain_free=True is incompatible with with_retain=True")
            print("Adding retain data")
            retain_split = "retain" + str(100 - int(split.split("_")[0].replace("forget", ""))).zfill(2)
            retain_train = load_dataset('locuslab/TOFU', retain_split)['train']
            #! Follow tofu, keep retain == forget count (unless caller passes
            #  retain_num_no_clamp=True, used for base-model full TOFU FT).
            if not kwargs.get('retain_num_no_clamp', False):
                retain_num = min(retain_num, len(base_forget_data))
            else:
                retain_num = min(retain_num, len(retain_train))
            retain_train = retain_train.select(
                range(len(retain_train) - retain_num, len(retain_train))
            )
            self.retain_length += len(retain_train)
            base_retain_data = datasets.concatenate_datasets([base_retain_data, retain_train])

        #! Augment forget data
        if expand_forget:
            print("Adding forget data")
            expand_qanum = kwargs.get('expand_qanum', 2)
            if expand_qanum > 0:
                expand_qa = collect_expand_data(
                    expand_qanum=expand_qanum, path=kwargs.get('paraphrase_path'),
                )
                tmpdata = datasets.Dataset.from_list([{'question': q, 'answer': a} for q, a in expand_qa])
            else:
                #! Otherwise we copy the original forget data
                tmpdata = load_dataset('locuslab/TOFU', split)['train']
            base_forget_data = datasets.concatenate_datasets([base_forget_data, tmpdata])
            self.forget_length += len(tmpdata)
            
        if with_perturb:
            print("Adding perturb data")
            perturb_qa = collect_perturb_data(
                expand_qanum=kwargs.get('expand_qanum', 3),
                path=kwargs.get('perturb_path')
            )
            tmpdata = datasets.Dataset.from_list([{'question': q, 'answer': a} for q, a in perturb_qa])
            self.retain_length += len(tmpdata)
            base_retain_data = datasets.concatenate_datasets([base_retain_data, tmpdata])

        # -------- F2D-DiD: explicit, balanced factorial contrasts --------
        if data_role in {'f2d_did_a1', 'f2d_did_a2'}:
            if counterfactual_path is None:
                raise ValueError(f"{data_role} requires counterfactual_path")
            if with_retain:
                raise ValueError(f"{data_role} must run with with_retain=False")

            units = load_ciru_units(counterfactual_path)
            ce_rows, uniform_rows = factorial_dual_roles(units, data_role)
            base_forget_data = datasets.Dataset.from_list(ce_rows)
            base_retain_data = datasets.Dataset.from_list(uniform_rows)
            self.forget_length = len(base_forget_data)
            self.retain_length = len(base_retain_data)
            ce_cell, uniform_cell = (
                ("C11", "C01") if data_role == 'f2d_did_a1'
                else ("C10", "C00")
            )
            print(
                f"Loaded balanced F2D-DiD {data_role}: "
                f"CE={ce_cell}({self.forget_length}), "
                f"uniform={uniform_cell}({self.retain_length}) from "
                f"{counterfactual_path}"
            )

        # -------- F2R: derive both roles from forget-conditioned data only --------
        elif data_role in {'f2r_a1', 'f2r_a2'}:
            if counterfactual_path is None:
                raise ValueError(f"{data_role} requires counterfactual_path")
            if with_retain:
                raise ValueError(f"{data_role} must run with with_retain=False")

            matched, mismatched, records = load_f2r_pairs(counterfactual_path)
            matched_data = datasets.Dataset.from_list(matched)
            mismatched_data = datasets.Dataset.from_list(mismatched)
            print(
                f"Loaded F2R supervision: {len(records)} matched and "
                f"{len(mismatched)} mismatched pairs from {counterfactual_path}"
            )

            # `forget` is the CE/remember role in remember+uniform; `retain`
            # is the KL-to-uniform role. These names come from upstream ULD
            # and do not imply access to the real retain set here.
            if data_role == 'f2r_a1':
                ce_data = datasets.concatenate_datasets([
                    base_forget_data, matched_data,
                ])
                uniform_data = datasets.concatenate_datasets([
                    base_retain_data, mismatched_data,
                ])
            else:
                ce_data = matched_data
                uniform_data = datasets.concatenate_datasets([
                    base_forget_data, base_retain_data, mismatched_data,
                ])

            base_forget_data = ce_data
            base_retain_data = uniform_data
            self.forget_length = len(ce_data)
            self.retain_length = len(uniform_data)

        # -------- Dual-ULD: repartition data based on R_sub / data_role --------
        elif data_role is not None:
            if r_sub_indices_path is None:
                raise ValueError("data_role is set but r_sub_indices_path is missing")
            with open(r_sub_indices_path) as f:
                rsub_info = json.load(f)
            rsub_idx = set(rsub_info["indices"])

            # Real retain rows are the first `retain_num` of base_retain_data;
            # perturb rows (if any) were appended after. retain_num was already
            # clamped above to min(retain_num, len(forget)).
            retain_train_count = retain_num if with_retain else 0

            real_retain = base_retain_data.select(range(retain_train_count))
            extra_retain = base_retain_data.select(range(retain_train_count, len(base_retain_data)))

            rsub_rows = real_retain.select([i for i in range(len(real_retain)) if i in rsub_idx])
            rfar_rows = real_retain.select([i for i in range(len(real_retain)) if i not in rsub_idx])

            if data_role == 'a1':
                # forget-role: original forget + R_sub.   retain-role: R_far + perturb.
                a1_forget = datasets.concatenate_datasets([base_forget_data, rsub_rows])
                a1_retain = datasets.concatenate_datasets([rfar_rows, extra_retain])
                base_forget_data = a1_forget
                base_retain_data = a1_retain
                self.forget_length = len(a1_forget)
                self.retain_length = len(a1_retain)
            elif data_role == 'a2':
                # forget-role: R_sub only. retain-role: original forget + R_far + perturb
                # (keep A2 quiet everywhere except R_sub via uniform regularization).
                a2_forget = rsub_rows
                a2_retain = datasets.concatenate_datasets([
                    base_forget_data, rfar_rows, extra_retain
                ])
                base_forget_data = a2_forget
                base_retain_data = a2_retain
                self.forget_length = len(a2_forget)
                self.retain_length = len(a2_retain)
            else:
                raise ValueError(f"Unknown data_role: {data_role}")

        base_forget_data = datasets.concatenate_datasets([
            base_forget_data, base_retain_data
        ])
        self.forget_data = base_forget_data
        self.eval_sets = {
            'forget': self.forget_eval,
            'perturb': self.perturb_eval,
            'paraphrase': self.paraphrase_eval,
        }
        if self.retain_eval is not None:
            self.eval_sets['retain'] = self.retain_eval
        print("In all ToFU Train: ", self.forget_length, self.retain_length)


def collect_expand_data(
    expand_qanum=10, path="data/aug_data/tofu/forget10_perturbed/paraphrase_res.csv",
):
    res = []
    import pandas as pd
    df = pd.read_csv(path)
    for idx, line in df.iterrows():
        para_question = list(set(eval(line.iloc[2])))
        para_answer = list(set(eval(line.iloc[3])))
        tmpres = []
        for para_q in para_question:
            for para_a in para_answer:
                tmpres.append((para_q, para_a))
        tmpres = tmpres[:expand_qanum]
        res.extend(tmpres)
    print("Expand num: ", len(res))
    return res

def collect_perturb_data(
    expand_qanum=10, path="data/aug_data/tofu/forget10_perturbed/perturb_res.csv",
):
    res = []
    import pandas as pd
    df = pd.read_csv(path)
    for idx, line in df.iterrows():
        para_question = line.iloc[2]
        para_answer = list(set(eval(line.iloc[3])))
        tmpres = []
        for para_a in para_answer:
            tmpres.append((para_question, para_a))
        tmpres = tmpres[:expand_qanum]
        res.extend(tmpres)
    print("Perturb num: ", len(res))
    return res
