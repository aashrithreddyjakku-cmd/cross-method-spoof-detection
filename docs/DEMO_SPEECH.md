# Demo speech (about 9 minutes)

Speakers follow the work packages: **Yash** (A, data), **Sejal** (B, models and training), **Aashrith** (C, evaluation,
explainability, demo). Swap parts if your roles differ. Every number below comes from our result tables.

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
We grouped the attacks into four families by how they generate the final waveform. Then we held out one whole family at a
time, so the model is tested on something truly new, with the same speakers and recording conditions.

One finding came before any training. The length of the silence at the start of a clip separates real from fake with an AUC
of about 0.71. That has nothing to do with speech, and a model can cheat with it. Keep that in mind.

Over to Sejal.

---

## 2. Sejal: models and training (about 1.5 min)

Thanks. We compared two kinds of model.

As baselines, we used classical MFCCs, which summarise each clip as a set of statistics, with logistic regression and with an
SVM.

Against those, we trained convolutional networks on the full spectrogram image. The main one is a Light CNN, or LCNN. Its
max-feature-map activations work like a built-in feature selector, which suits this task because only some spectral regions
carry evidence. We also trained a smaller plain CNN.

Fake clips outnumber real ones in the data, so we used class-weighted loss. We used SpecAugment to mask random stripes of the
spectrogram, and we stopped early based on validation error, never on test data.

Our main metric is Equal Error Rate, the point where false alarms and misses are equal. Lower is better.

Cost matters too, because a detector should be able to run on a live call. The LCNN has about 164 thousand parameters and
processes a 4-second clip in about 38 milliseconds on a single CPU thread, so it is far faster than real time. The models
were trained on a CPU for a limited number of epochs with one seed, so these are early numbers.

Aashrith will show what happened.

---

## 3. Aashrith: results (about 2 min)

*[Show `report_per_attack.png`.]*

On the official protocol, where every test attack except two re-built ones is new to the model, the LCNN reached **11 percent
EER**. The SVM reached 13, the small CNN about 15, and logistic regression 20. The LCNN's margin over the SVM is small: their
confidence intervals overlap.

The overall number hides the real story. For 9 of the 13 attacks, the LCNN's error is under 2 percent, including vocoders it
never saw. Most of the error comes from three attacks, A17, A18 and A19, at between 21 and 30 percent. All three are voice
conversion systems that modify a real human recording, so most of the signal really is human.

*[Show `report_lofo.png`.]*

Holding out whole families shows this clearly. With waveform modification held out, the LCNN goes from about 2 percent on
families it knows to **almost 35 percent** on the unseen one. The simpler SVM gets 29 percent there. So the model that was
better on the benchmark did worse on the unseen family, though this is one seed. With neural vocoders held out, the unseen
error is only about 2 percent. So being new is not what makes an attack hard. The type of attack is what matters.

*[Show `report_roc.png`.]*

The largest failure is on real-world audio. On In-the-Wild, the LCNN's EER rises to **39 percent**, and it flags 81 percent of
genuine clips as fake. It seems to have learned that clean studio speech is real and that anything else is suspicious.

Now the silence finding. When we removed silence for the SVM, its error on families it knew roughly doubled. So on this
benchmark, part of its performance came from silence, not speech.

---

## 4. Aashrith: live demo (about 2.5 min)

*[Open the app. Slider at 0.10.]*

This is our detector. I upload a clip, and it returns a verdict, the spectrogram, and two explanations. I chose these clips
from our test results to show both the successes and the failures, so they do not represent overall accuracy.

**Clip 1.** *[Analyse.]* This is real speech. The model says bona fide, with a spoof probability of about 0.02.

**Clip 2.** *[Analyse.]* This is text-to-speech using Griffin-Lim, a generator never used in training. The verdict is spoof,
at 0.98. The middle plot, Grad-CAM, shows where the evidence for "spoof" sits. The right plot shows how much the score drops
when we hide each frequency band.

**Clip 3.** *[Analyse.]* A neural vocoder, WaveRNN, also never seen in training. Again spoof, at 0.98.

**Clip 4.** *[Analyse.]* This one is fake: voice conversion applied to a real recording. The model says **real**, with high
confidence. This is the failure we described.

**Clip 5.** *[Analyse. Do not play it.]* This is a genuine recording from In-the-Wild. We won't play it, because it is a real
person's voice. The model calls it **spoof**, at 0.98. This is the false alarm problem.

So the system can be very confident and still be wrong. For anything like fraud detection, that is the most important lesson.

Back to Yash.

---

## 5. Yash: what it means (about 1 min)

To return to the gap in our survey. First, with speakers and channel controlled, the generalisation gap was not spread evenly.
It was concentrated in attacks that modify real speech. Second, benchmark rank did not predict robustness on that unseen
family. Third, preprocessing changes what a model learns, as the silence result shows. And fourth, per-attack reporting
showed that a few attacks cause most of the error, which a single EER hides.

We should be clear about the limits. This is a subset of the data with one seed, and some held-out groups are small. In
ASVspoof, attacks are also tied to particular speaker pools, and no split can fully remove that. We have not yet trained the
larger architectures or run the full leave-one-method-out study.

Next, we would train with noisy and compressed audio and with waveform-modification examples, remove silence during training,
and repeat on the full dataset with several seeds.

Thank you. We are happy to take questions.

---

## Likely questions

**Why is your EER so much worse than the published 0.07 to 2.9 percent?**
We trained on a small subset, for few epochs, on a CPU. Those published figures are also mostly in-domain. Our focus was the
controlled comparison, not a record score.

**What is EER?**
The error rate at the threshold where the false alarm rate equals the miss rate. It does not depend on choosing a threshold,
which helps with imbalanced data.

**Why not just use accuracy?**
In the full training set, about 90 percent of clips are fake, so always answering "fake" scores about 90 percent.

**Why did the neural-vocoder fold have a lower error on unseen than on seen attacks?**
Its "seen" test set includes the hard waveform-modification attacks, while the unseen set contains only vocoders, which are
easier to detect. That model also had not converged (18.5 percent validation EER), so it isn't our strongest evidence.

**How do you know there is no data leakage?**
`check_splits.py` confirms there is no shared speaker or utterance across train, validation and test, and that the held-out
family is absent from training and validation. A16 and A19 are folded into A04 and A06 because they use the same algorithms.

**Why is A19 hard when it uses the same algorithm as the training attack A06?**
It was built on different speakers and data, so the model probably learned some speaker- or data-specific cues as well as the
algorithm. We have not tested that explanation directly.

**What does the explainability show?**
On 25 to 40 clips per attack, it is exploratory. With neural vocoders held out, hiding any single band barely moves the score,
so the evidence looks spread out. With waveform modification held out, the bands above about 1.7 kHz mattered most for A17
and A13. A06 scored low to begin with, so there was little evidence to remove.

**Can I try my own voice?**
You can, but a laptop microphone is noisy and the model has high false alarms on noisy audio, so it may well call you fake.
We have not tested the microphone route.

**Did you train ResNet-18 and VGGish?**
Not yet. We profiled them for size and speed only.

**Was everything run on the full dataset?**
No, on a seeded, balanced subset of 3.4 percent of ASVspoof 2019 LA. The pipeline itself works the same on the full data.
