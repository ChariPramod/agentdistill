# Terms of service: before you train on a teacher's outputs

**Read this before running `agentdistill train`.** It is not legal advice, and it is not a substitute for reading
your own contract.

## The short version

Provider terms of service govern whether you may use a model's outputs to train another model. Several frontier
providers restrict using their outputs to build competing models.

Distilling your own agent's traces, from your own account, to reduce your own inference costs, is a different
case from building a competing product. But it is **not automatically permitted**, and the answer differs by
provider, by plan, and by negotiated contract.

## What to check

1. **The provider's terms for the specific model that produced your traces**, as of the date you trained. Terms
   change; the version that applied when you collected the traces may not be the version that applies now.
2. **Your enterprise agreement, if you have one.** It usually supersedes the public terms, sometimes in your
   favour.
3. **Whether "competing model" is defined**, and whether an internal cost-reduction student falls inside it.
4. **Whether the output of the student is itself restricted** — some terms reach downstream.
5. **Data residency and customer data.** Your traces contain your users' data. Training a model on them, and
   where that model then runs, may be governed by agreements entirely separate from the model provider's.

## Why this project supports open-weight teachers

A large open-weight model as teacher and a small open-weight model as student is a first-class path here, not a
fallback. If your teacher's terms forbid distillation, or you would rather not have the question hanging over the
project, you can run the entire pipeline on models whose licenses permit it.

Check the student's license too. "Open weights" is not one license, and several popular ones carry acceptable-use
terms or naming requirements that survive fine-tuning.

## What this tool does and does not do

- It **does** record `teacher_model` on every trace, so you can restrict a dataset to one teacher
  (`curate.filters: [teacher]` with `curate.teacher_models`), and prove afterwards which teacher a dataset came
  from.
- It **does** record the full lineage: which traces, which filters, which base model, which dataset hash.
- It **does not** ship a default teacher, and it will not silently call one.
- It **does not** check your terms for you. Nothing in this tool constitutes permission.

## If you are publishing results

Say which teacher produced the traces and under what terms you believe the training was permitted. The benchmark
post that accompanies this project does the same. A cost-reduction claim that quietly skips this is not a claim
anyone should build on.
