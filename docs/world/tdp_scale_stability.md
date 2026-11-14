# TDP scale revision (2026-11-14)

Forward flow: image/action/reward embeddings plus positional tokens → pre-RMS perceptual block → pre-RMS memory block → observation, immediate reward, action-value predictions. Candidate actions read the same carry; the selected action and observed reward produce the committed trace update. Only traces and W cross steps. Training uses 32-step truncated BPTT and carries states across parameter updates.

The previous rule normalized Q/K, but left V unbounded, accumulated delta innovations through complex traces with sqrt(1-rho²) input gain, used unrestricted complex mixture coefficients, and gave long-memory writes and erases different gains. These leave both target scale and erase contraction uncontrolled. Post-residual RMS alone does not bound state.

Changes:

- Pre-RMS before Q/K/V projection; Q/K retain L2 normalization. V uses tanh. Current-token outer products are averaged over token count.
- Complex traces use rho*rotation and input gain 1-rho. Rates are constrained to [1/512,1]; each token key trace has complex Euclidean norm at most 1, and value trace at most sqrt(head dimension), from zero initialization. Short delta innovations use tanh(v - prediction), giving a bounded input even with feedback.
- Positive softmax weights replace unconstrained complex band mixing. Cross moment C and key Gram G share denominator max(1, trace(raw G)). G is PSD with trace <= 1. Phase remains in complex trace rotation, not in signed erase weights.
- eta = 1-exp(-softplus(log_kappa)*abs(RPE)) lies in [0,1]. W += eta*(1[RPE>0]*C - W G). The same learning rate scales writing and erasing. Negative surprises erase without writing. I-eta G is non-expansive; this alone is not a proof of a uniform bound under changing targets/keys.
- Project each head W onto Frobenius radius D (head dimension). This guarantees W RMS <= 1 even across changing learned representations. Radius D permits D orthogonal unit keys with value norm sqrt(D). No global long-memory decay is introduced. Wclip reports the proportion of heads whose unprojected update crossed the radius.
- Log eta, effective kappa, value-trace RMS Vn and Wclip in addition to Mn/Wn. Existing tr_ metrics are from the final training chunk, not evaluation-wide state statistics.

Tradeoffs: bounded values and averaged moments can reduce signal strength; positive mixtures remove signed phase readout; projection limits capacity if frequently active. Stability is enforced, but learning quality requires the new run. Existing checkpoint arrays retain their shapes but have changed semantics; this experiment starts from scratch. Old retention diagnostic assumes the old formula and must not be used to interpret the new traces.

Validation: CPU 512-step adversarial rollout (100x value projection and band logits, kappa parameter 20, near-unit decay), state bounds, finite recurrent gradients, and gate=False zero-state check. Tests live in tests/world/test_tdp_stability.py; invoked directly because pytest is absent from /venv/jax.

Run: supervisor program rltdp-stable, 960 iterations, B=40, L=32, eval_B=1024, seed=0. Outputs: runs/world/stable_20261114/tdp_stable.jsonl and checkpoint arrays. Logs: /var/log/portal/rltdp-stable.log. Previous run outputs are preserved. No automatic restart after completion or failure.
