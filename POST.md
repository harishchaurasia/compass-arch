# LinkedIn post - Compass

Primary image: `analysis/figures/headline_metrics.png`

---

AI agents have a big problem: they're overconfident. 😤

Ask an agent if it's sure, it says "100%," then cancels the wrong order, refunds the wrong customer, or loops forever insisting it's got it - while being completely wrong. The confidence is real to the model. It's just disconnected from reality. And in production, that's the difference between a helpful assistant and a very expensive mistake. 💸

So I'm building Compass 🧭 - my take at fixing this.

Compass is a calibration layer that wraps any AI agent and decides, before every action or tool-call, whether to execute, self-verify, or abstain - based on how trustworthy the agent's own confidence really is.

The core idea 💡: confidence is earned, not announced.
 • A bare "I'm 100% sure" gets discounted toward a realistic baseline before the agent can act.
 • The longer it loops without real progress, the more we shave that confidence down, until Compass makes it stop, verify, or hand off. 🛑

Here's the eval part - testing it. 🧪
I ran Compass on 115 real customer-service tasks built upon τ-bench (tau-bench) across two very different models:

🔴 On GPT-4o-mini, a plain agent took a destructive, irreversible action while wrong on 63 of 115 tasks (55%). Compass cut that by exactly two-thirds, down to 21.

🟢 On an overconfident local model (Qwen2.5 14B), where the model's confidence carries no signal at all, a base-rate prior drove destructive actions to 0 on this benchmark.

The thesis 🎯: calibration works when a model's confidence carries signal, and needs a base-rate prior when it doesn't. Not "solved" - one benchmark, two models - but the failure mode that makes agents dangerous in production is now gated. ✅

Why it matters: Every prevented mistake is a refund not issued, a cleanup avoided, a doomed run stopped before it burns more compute. Reliability is cost savings, measured in avoided damage. 💰

Heading towards production, built in the open 🛠️. Next phase is extending the benchmark to Qwen2.5 7B and Llama 3.1 8B, then closing the open question of recovering coverage lost to caution.

I'll keep sharing the progress here. If agent safety, security, robustness, and reliability is your world, let's connect. PRs welcome. 🤝

#AI #MachineLearning #AIAgents #LLM #AICalibration #AISafety #evals #AIEvals #evaluationframeworks

---

## Optional carousel captions (if uploading all three images)

1. **headline_metrics.png** - The headline: Compass trades some task success for far fewer destructive failures, across two models.
2. **outcome_categories.png** - Task by task: where Compass wins, where it over-abstains, and where both agents struggle (gpt-4o-mini).
3. **threshold_sensitivity.png** - The open problem: even tuning the threshold, abstention rarely fires before the first destructive action. This is what the next phase tackles.
