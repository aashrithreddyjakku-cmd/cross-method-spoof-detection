# Demo speech (about 9 minutes)

Speakers follow the work packages: **Yash** (A, data), **Sejal** (B, models and training), **Aashrith** (C, evaluation,
explainability, demo). Swap parts if your roles differ. Every number below comes from our result tables in `results\tables\`.

---

## 1. Yash: the problem and the data (about 2.5 min)

Good morning. We are Yash, Sejal and Aashrith, and our project is cross-method synthetic speech detection using CNNs on
mel-spectrograms.

Voice cloning is now cheap, and it is already being used for impersonation scams and fake emergency calls. Published detectors
often report error rates under three percent, so this can look like a solved problem. But our literature survey found that
those numbers come from testing on data very similar to the training data. When the synthesis method or the recording
conditions change, detectors collapse.

So we did not set out to build just another detector. We asked: **how badly does a CNN fail on a voice generator it has never
seen, and why?**

I handled the data. We used ASVspoof 2019 Logical Access, which has 19 attack systems, for training and testing. We used
In-the-Wild, which is real deepfakes collected from the internet, purely as an outside test. Our laptops could not process
the full 15 gigabytes, so we built a balanced, reproducible subset: 4,180 ASVspoof clips and 600 In-the-Wild clips. It still
contains all 107 speakers and all 19 attacks.

Every clip is cut or looped to 4 seconds and converted to a log-mel spectrogram: an image with time across, frequency up,
and brightness for energy.

We also made the comparison fair. No speaker appears in both training and testing, and a script checks this automatically.
We grouped the attacks into four families by how they generate the final waveform. Then we held out one whole family, or one
single attack, at a time, so the model is tested on something truly new, with the same speakers and recording conditions.

One finding came before any training. The length of the silence at the start of a clip separates real from fake with an AUC
of about 0.71. That has nothing to do with speech, and a model can cheat with it. Keep that in mind.

Over to Sejal.

---

## 2. Sejal: models and training (about 1.5 min)

Thanks. We compared two kinds of model.

As baselines, we used classical MFCCs, which summarise each clip as a set of statistics, with logistic regression and with an
SVM.

Against those, we trained four convolutional networks on the full spectrogram image: a small plain CNN, a Light CNN, a
ResNet-18, and a VGG-style network with CBAM attention, which learns which channels and regions to focus on.

Fake clips outnumber real ones in the data, so we used class-weighted loss. We used SpecAugment to mask random stripes of the
spectrogram, and we stopped early based on validation error, never on test data. Each CNN on the official protocol was trained
with three random seeds, and we report the mean.

Our main metric is Equal Error Rate, the point where false alarms and misses are equal. Lower is better.

In total we finished 116 runs, on a free Kaggle GPU, in about an hour and a half. We chose VGGish-CBAM as our main model
because it had the lowest validation error, 1.15 percent. That choice was made on validation data only.

Cost matters too, because a detector should be able to run on a live call. VGGish-CBAM has about 298 thousand parameters and
processes a 4-second clip in about 13 milliseconds on a laptop CPU, so it is far faster than real time. ResNet-18 has about
2.8 million.

Aashrith will show what happened.

---

## 3. Aashrith: results (about 2 min)

*[Show `per_method_p0_vggish_cbam_mel64_w400_s1_test.png`.]*

On the official protocol, where every test attack except two re-built ones is new to the model, the small CNN and VGGish-CBAM
both reached **7 percent EER**. They are effectively tied. The Light CNN got about 9, ResNet-18 about 12, the SVM 13, and
logistic regression 20.

The overall number hides the real story. Across three seeds, VGGish-CBAM's error is under 4 percent on 9 of the 13 test attacks.
One attack, **A17, sits at 28 percent**. A17 is voice conversion that filters a real human recording, so most of the signal
really is human.

*[Show `gap_p1_lofo.png`.]*

Holding out whole families shows this clearly. With waveform modification held out, VGGish-CBAM goes from under 1 percent on
families it knows to **28 percent** on the unseen one. The small CNN gets 30 and the plain SVM gets 29. So on this family, the
deep models were no better than the classical baseline. With neural vocoders or Griffin-Lim held out, there is almost no gap. So
being new is not what makes an attack hard. The type of attack is what matters.

*[Show `lomo_heatmap.png`.]*

We then held out one attack at a time, for all 17 attack groups. For 13 of them, the unseen error stays under 4 percent. The
exceptions are A06 at 47 percent, A17 at 24, A18 at 11 and A04 at about 5. These folds are small, between 25 and 80 fake clips,
so treat the exact values with care.

*[Show `roc_det_p0_itw.png`. This plot shows one seed per model.]*

The largest failure is on real-world audio. On In-the-Wild, the mean EER rises to **44 percent** for VGGish-CBAM and **55 percent**
for the small CNN, which is worse than guessing. Those two tied on the benchmark. ResNet-18 was the weakest CNN on the benchmark
and the best here, at 34 percent. So benchmark rank did not predict real-world performance. Even when we trained VGGish-CBAM on
all attack types, In-the-Wild error was still 33 percent.

*[Show `ablation.png`.]*

Now the silence finding. Removing silence raised the SVM's error on families it knew from around 10 percent to around 25. For
VGGish-CBAM, averaged over three folds, unseen error rose from 15 percent to 24. So part of the performance on this benchmark
depends on silence, not speech. The other settings we tested, like mel bands, window size and normalisation, moved the result
far less than the choice of held-out family did, and those runs used one seed.

---

## 4. Aashrith: live demo (about 2.5 min)

*[Open the app. Slider at 0.22.]*

This is our detector, running VGGish-CBAM on this laptop. I upload a clip, and it returns a verdict, the spectrogram, and two
explanations. I chose these clips from our test results to show both the successes and the failures, so they do not represent
overall accuracy.

**Clip 1.** *[Analyse.]* This is real speech. The model says bona fide, with a spoof probability of about 0.02.

**Clip 2.** *[Analyse.]* This is text-to-speech using Griffin-Lim, a generator never used in training. The verdict is spoof,
at 0.98. The middle plot, Grad-CAM, shows where the evidence for "spoof" sits. The right plot shows how much the score drops
when we hide each frequency band.

**Clip 3.** *[Analyse.]* A neural vocoder, WaveRNN, also never seen in training. Again spoof, at 0.98.

**Clip 4.** *[Analyse.]* This one is fake: voice conversion applied to a real recording. The model says **real**, with a score
of 0.02. This is the failure we described.

**Clip 5.** *[Analyse. Do not play it.]* This is a genuine recording from In-the-Wild. We won't play it, because it is a real
person's voice. The model calls it **spoof**, at 0.98. This is the false alarm problem. This model flags about 90 percent of
genuine In-the-Wild clips as fake.

So the system can be very confident and still be wrong. For anything like fraud detection, that is the most important lesson.

Back to Yash.

---

## 5. Yash: what it means (about 1 min)

To return to the gap in our survey. First, with speakers and channel controlled, the generalisation gap was not spread evenly.
It was concentrated in a few attacks, especially those that modify real speech. Second, benchmark rank did not predict real-world
robustness, and on the hardest family the deep models did no better than an SVM. Third, preprocessing changes what a model
learns, as the silence result shows. And fourth, per-attack reporting showed that a few attacks cause most of the error, which a
single EER hides.

We should be clear about the limits. This is a 3.4 percent subset of ASVspoof, and some held-out groups have only 25 fake clips.
Results varied between seeds, especially on real-world audio, and some runs used one seed. In ASVspoof, attacks are also tied to
particular speaker pools, and no split can fully remove that. Our explanation plots are exploratory, as the two methods often
disagree.

Next, we would train with noisy and compressed audio and with waveform-modification examples, remove silence during training,
and repeat on the full dataset.

Thank you. We are happy to take questions.

---

## Likely questions

**Why is your EER so much worse than the published 0.07 to 2.9 percent?**
We trained on a small subset. Those published figures are also mostly in-domain. Our focus was the controlled comparison, not a
record score. Trained on all attack types, tested on those same types, VGGish-CBAM reached 1.8 percent.

**Why is validation 1.15 percent but test 7 percent?**
On the official protocol, validation contains only attacks the model trained on. The test set contains mostly new ones.

**Why choose VGGish-CBAM if the small CNN has the same test EER?**
We choose on validation data so that the test set stays untouched. VGGish had the lowest validation error, and the test difference
is within the spread between seeds. The results do not show that it is better than the small CNN on the test set.

**What is EER?**
The error rate at the threshold where the false alarm rate equals the miss rate. It does not depend on choosing a threshold,
which helps with imbalanced data.

**Why not just use accuracy?**
In the full training set, about 90 percent of clips are fake, so always answering "fake" scores about 90 percent.

**Why does an unseen family sometimes have lower error than seen ones?**
Griffin-Lim and neural vocoders are easy to detect. The "seen" test set includes harder attacks, such as those that modify real
speech, so it can have the higher error.

**Why is A06 so high when it is held out, at 47 percent?**
In our data A19 is folded into A06, because they use the same algorithm. When that algorithm is absent from training, the model
scores those clips about as low as real speech (mean score 0.03). When A06 is in training, on the official protocol, the error
is about 8 percent. The two protocols differ, so this comparison is only suggestive.

**How do you know there is no data leakage?**
`check_splits.py` confirms there is no shared speaker or utterance across train, validation and test, and that the held-out
method is absent from training and validation. We also trained with shuffled labels, and the error stayed near chance: about 48
percent on average.

**What does the explainability show?**
Mostly that we should be careful. Hiding a single frequency band usually barely moves the score, so the evidence looks spread out.
Grad-CAM and band occlusion often disagree: their correlation ranged from strongly negative to strongly positive across folds. In
most folds, Grad-CAM changed when we randomised the model's weights, as it should, but in the worst fold the correlation was 0.68.
So we treat these plots as exploratory, not as proof of what the model learned.

**Is 3 seeds enough?**
It shows the spread, but only roughly. On In-the-Wild, the VGGish-CBAM seeds ranged from 36 to 52 percent, and the demo uses seed 1,
which is the 52. We fixed seed 1 in advance and did not choose it by test results.

**Can I try my own voice?**
You can, but a laptop microphone is noisy and the model has high false alarms on noisy audio, so it may well call you fake.
We have not tested the microphone route.

**Was everything run on the full dataset?**
No, on a seeded, balanced subset of 3.4 percent of ASVspoof 2019 LA. The pipeline itself works the same on the full data.
