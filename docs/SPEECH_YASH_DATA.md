# Yash: data and features speech (about 4 minutes)

Good morning. I'm Yash, and I was responsible for the data and features: everything the models learn from and are tested on.

Our question was how well a detector handles a voice generator it has never seen. That can only be answered if the data is set
up carefully, so I'll cover four things: the datasets, how we turned audio into features, how we built fair tests, and a problem
we found in the data before training anything.

**The datasets.** We used two. ASVspoof 2019 Logical Access has real speech and 19 different attack systems, and we used it for
training and testing. In-the-Wild has real and fake audio collected from the internet, and we used it only as an outside test.
It was never used for training, validation or choosing a threshold.

Together they are about 15 gigabytes and 153 thousand files, which our laptops could not handle. So we built a subset. Instead of
downloading everything, a script reads the index of each zip file and fetches only the clips it selects. We ended up with 4,180
ASVspoof clips, which is 3.4 percent, and 600 In-the-Wild clips. The sampling is seeded, so anyone can rebuild the same files. It
also samples evenly across speakers and attacks, so all 107 speakers and all 19 attacks are still there.

**The features.** Every clip is converted to 16 kilohertz and fixed to exactly 4 seconds. Longer clips are cut, and shorter ones
are repeated. We then compute a log-mel spectrogram, using a 25 millisecond window, a 10 millisecond step and 64 mel bands. Each
clip becomes an image of 64 by 401: time across, frequency up, and brightness for energy. We also built six other versions, with
different numbers of bands, different window sizes, and silence removed, so we could test how much those choices matter.

**Fair tests.** This is the most important part of my work. We created four kinds of split.

The first is the official protocol, where most test attacks are not in training. The second trains and tests on all attacks, as a
best-case reference. The third holds out one attack at a time, and the fourth holds out a whole family of attacks.

We defined four families by what finally produces the waveform: neural vocoders, source-filter vocoders, Griffin-Lim, and waveform
modification. I checked every attack against the table in the ASVspoof paper. Two attacks needed special care: A16 uses the same
algorithm as A04, and A19 the same as A06. We treat each pair as one method. Otherwise, holding out A04 would still leave its twin
in training, and the method would leak. That leaves 17 distinct methods.

In our own splits, speakers are divided 64, 16 and 27 between training, validation and testing, and no speaker appears in more than
one. A script called `check_splits` verifies this automatically, along with the rule that a held-out method never appears in
training or validation. All of those checks pass.

**The problem we found.** Before any model was trained, I tested simple cues. In ASVspoof, real clips start with about one second of
silence on average, and fake clips with about 0.6 seconds. That silence length alone separates real from fake with an AUC of about
0.71. It has nothing to do with the voice, and a model could cheat with it. On In-the-Wild, the same cue gives 0.50, which is pure
chance. So it doesn't carry over to real-world audio.

That finding shaped the experiments. It's why we built the version with silence removed. When we later trained on it, our main
model's error on unseen families rose from 15 percent to 24 percent, so part of the performance on this benchmark does depend on
silence.

**The limits.** I want to be clear about these. This is a small subset, and some single-attack tests have only 25 fake clips, so one
clip moves the error by 4 percent. That is why we trust the family-level results more. In ASVspoof, each attack is also tied to a
particular pool of speakers, and no split can fully remove that. Repeating short clips adds joins that aren't in the original audio.
And the data is English only.

In the end, the team got seven feature sets and all the splits through one shared, private Kaggle dataset, with no raw audio in it,
to respect the dataset licences. The full pipeline can be rebuilt with one script, using the same seed.

Thank you. I'll now hand over to Sejal, who will explain the models.

---

## If they ask

**Why these four families?**
A CNN on spectrograms mostly sees artefacts from the stage that produces the waveform. The families are our design choice, not a
property of the data. A07, A13 and A18 are borderline, and we recorded why we placed them where we did in `docs/ATTACKS.md`.

**Why is the subset 43 percent real when the full data is about 90 percent fake?**
We sampled about evenly so every attack and speaker is represented. Our training still uses a class-weighted loss.

**Does the small subset make the results wrong?**
It makes them noisier. The absolute numbers won't match published ones, so we report confidence intervals and compare families
rather than single attacks.

**How do you know the subset is reproducible?**
The same seed gives the same files, and we store checksums for the manifest, the splits and the main feature file.
