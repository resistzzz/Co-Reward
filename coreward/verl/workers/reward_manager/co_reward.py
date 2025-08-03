# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict, Counter

import torch

from verl import DataProto
from verl.utils.reward_score import _default_compute_score


def majority_vote(ans_list, empty_value=''):
    ans_list = [a for a in ans_list if a is not None and str(a).strip() != '']
    if not ans_list:
        return empty_value  # 或 return 0, 或 return '[NO_ANSWER]'
    return Counter(ans_list).most_common(1)[0][0]



class CoRewardManager:
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source") -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score
        self.reward_fn_key = reward_fn_key

    def _extract_valid_response_str(self, item):
        prompt_ids = item.batch["prompts"]
        prompt_length = prompt_ids.shape[-1]
        valid_prompt_length = item.batch["attention_mask"][:prompt_length].sum()
        response_ids = item.batch["responses"]
        valid_response_length = item.batch["attention_mask"][prompt_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]
        response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
        return response_str

    def __call__(self, data_ori: DataProto, data_aug: DataProto, return_dict=False):
        """We will expand this function gradually based on the available datasets"""
        assert len(data_ori) == len(data_aug), "The original and augmented data must have the same length."
        ori_uid2answers = defaultdict(list)
        ori_all_answers = []
        for i, item in enumerate(data_ori):
            uid = item.non_tensor_batch["uid"]
            ans = self.extract_answer(self._extract_valid_response_str(item))
            ori_uid2answers[uid].append(ans)
            ori_all_answers.append(ans)
        aug_uid2answers = defaultdict(list)
        aug_all_answers = []
        for j, item in enumerate(data_aug):
            uid = item.non_tensor_batch["uid"]
            ans = self.extract_answer(self._extract_valid_response_str(item))
            aug_uid2answers[uid].append(ans)
            aug_all_answers.append(ans)
        
        ori_uid2pseudo = {uid: majority_vote(ans_list, empty_value='') for uid, ans_list in aug_uid2answers.items()}
        aug_uid2pseudo = {uid: majority_vote(ans_list, empty_value='') for uid, ans_list in ori_uid2answers.items()}
 
        reward_tensor_ori = torch.zeros_like(data_ori.batch["responses"], dtype=torch.float32)
        reward_tensor_aug = torch.zeros_like(data_aug.batch["responses"], dtype=torch.float32)

        N = len(data_ori) 
        for i in range(N):
            item = data_ori[i]
            ans = ori_all_answers[i]
            pseudo_label = ori_uid2pseudo[item.non_tensor_batch["uid"]]
            valid_response_length = item.batch["attention_mask"][item.batch["prompts"].shape[-1]:].sum().item()
            reward = 1.0 if ans == pseudo_label and pseudo_label != '' else 0.0
            if valid_response_length > 0:
                reward_tensor_ori[i, valid_response_length - 1] = reward
            
            item = data_aug[i]
            ans = aug_all_answers[i]
            pseudo_label = aug_uid2pseudo[item.non_tensor_batch["uid"]]
            valid_response_length = item.batch["attention_mask"][item.batch["prompts"].shape[-1]:].sum().item()
            reward = 1.0 if ans == pseudo_label and pseudo_label != '' else 0.0
            if valid_response_length > 0:
                reward_tensor_aug[i, valid_response_length - 1] = reward
            
        return reward_tensor_ori, reward_tensor_aug

    def extract_answer(self, solution_str: str):
        from verl.utils.reward_score.math import last_boxed_only_string, remove_boxed
        answer = ''
        try:
            string_in_last_boxed = last_boxed_only_string(solution_str)
            if string_in_last_boxed is not None:
                answer = remove_boxed(string_in_last_boxed)
        except Exception as e:
            print(e)
        return answer
