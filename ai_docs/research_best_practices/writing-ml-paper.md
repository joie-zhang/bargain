Highly Opinionated Advice on How to Write ML Papers
by Neel Nanda
11th May 2025
AI Alignment Forum
TL;DR
The essence of an ideal paper is the narrative: a short, rigorous and evidence-based technical story you tell, with a takeaway the readers care about
What? A narrative is fundamentally about a contribution to our body of knowledge: one to three specific novel claims that fit within a cohesive theme
Why? You need rigorous empirical evidence that convincingly supports your claims
So what? Why should the reader care?
What is the motivation, the problem you’re trying to solve, the way it all fits in the bigger picture?
What is the impact? Why does your takeaway matter? The north star of a paper is ensuring the reader understands and remembers the narrative, and believes that the paper’s evidence supports it
The first step is to compress your research into these claims.
The paper must clearly motivate these claims, explain them on an intuitive and technical level, and contextualise what’s novel in terms of the prior literature
This is the role of the abstract & introduction
Experimental Evidence: This is absolutely crucial to get right and aggressively red-team, it’s how you resist the temptation of elegant but false narratives.
Quality > Quantity: find compelling experiments, not a ton of vaguely relevant ones.
The experiments and results must be explained in full technical detail - start high-level in the intro/abstract, show results in figures, and get increasingly detailed in the main body and appendix.
Ensure researchers can check your work - provide sufficient detail to be replicated
Define key terms and techniques - readers have less context than you think.
Write iteratively: Write abstract -> bullet point outline -> introduction -> first full draft -> repeat
Get feedback and reflect after each stage
Spend comparable amounts of time on each of: the abstract, the intro, the figures, and everything else - they have about the same number_of_readers * time_to_read
Inform, not persuade: Avoid the trap of overclaiming or ignoring limitations. Scientific integrity may get you less hype, but gains respect from the researchers who matter.
Precision, not obfuscation: Use jargon where needed to precisely state your point, but not for the sake of sounding smart. Use simple language wherever possible.


Case study: The abstract of refusal is mediated by a single direction, broken down into the purpose of each sentence

Introduction
Your research only matters if people read, understand, and build upon it. This means that writing a good paper is a critical part of the research process. Further, the process of writing forces you to clarify your own thinking in ways that often reveal gaps or new insights - I’ve often only properly understood an idea after writing it up. Yet, to many, writing feels less fun than research and is treated as an after thought - a common but critical mistake.

In my experience supervising 20+ papers and reading/appreciating/being annoyed by a bunch more, I've developed my own opinionated framework for what I think makes a paper good and how to approach the writing process. I try to lay this out in this post, along with a bunch of concrete advice.[1] This post assumes you’ve already done a bunch of technical research, and focuses on how to effectively share it with the world, see my other posts for advice on the research part.

Caveat: I mostly have experience with writing mechanistic interpretability papers and this advice is written with that flavour. I expect much of it to generalise to the rest of ML and some to generalise to other fields but it's hard for me to say. Further, this is very much my personal opinionated, and optimised more for truth-seeking than getting into conferences[2]. See other great advice here and here.

Caveat 2: There are many reasonable objections to the academic paper as the format for communicating research. Alas, engaging with those is outside the scope of this post, it’s long enough as it is.

The Essence of a Paper
At its core, a paper should present a narrative of one to three specific concrete claims that you believe to be true, that build to some useful takeaway(s). Everything else in the paper exists to support this narrative. The second pillar of the paper is rigorous evidence for why they are true - obviously there will always be some chance you’re wrong, but they should be compelling and believable, without obvious glaring flaws.

Communicate the key idea
Motivate why someone should care about them
Contextualize them in existing literature
Communicate them precisely, with all relevant technical detail, terminology and background context
Provide sufficient evidence to support them
Crafting a Narrative
One of the critical steps that can make or break a paper is crafting a narrative. What does this actually mean? And how can you do it?

Research is about discovering new things and pushing forward our frontier of existing knowledge. I view a paper as something that finds insight and provides compelling evidence behind it. The way to tell when you could start writing a paper is when you have learned something insightful, in a way that could be made legible to someone else.

This is far easier said than done. A research project is often a mess of fun results, confusions, insights, and remaining mysteries. Even when you’ve made enough progress to write it up, you will likely have a great deal of tacit knowledge, interesting rabbit holes, dangling threads, etc - projects rarely feel done.

I find that converting a project into a great narrative is a subtle and difficult skill, one of the many facets of research taste. It’s something I’ve gotten much, much better at over time, and it’s hard to say what the best way to get better at it is, beyond experience. One exercise I’d recommend is taking papers you know, and trying to write down what their narrative is, and ask yourself what its strengths and weaknesses are. If at all possible, consult mentors/more experienced researchers for advice. But if you need to come up with one yourself, the right questions to ask look like:

Which of these results would be most exciting to show someone?
Actually show someone your findings and ask what they're most interested in
What seems particularly important?
Why should anyone care about this work?
What was hard about what you did, that perhaps no one else has done?
A good, compelling narrative comes with motivation and impact. The key points to be sure to cover:

The context of your insight
The problem you're trying to solve
Why this matters
What you have shown
Why the reader should believe it
What the insight is
Why do you need this kind of compressed narrative? Often there’s far more insight in a research project than can be contained in this structure. But it is impossible to convey this level of nuance in a paper. Readers will rarely take away more than a few sentences of content. Choose those sentences carefully. These are the insights shared by your paper, your contribution to the literature. These are the specific, concrete claims that you want to communicate - you cannot reliably communicate much more. You will need to compress your research findings down into a handful of claims, prioritise those, and accept that you may need to drop a bunch of other detail, or move it to appendices. If you don’t deliberately de-prioritise some details, then something else will get dropped, which may have been far more important.

What do I mean by claims? For example:

"Method X is the best approach on task Y (according to metric Z)"
"A substantial part of the model's behavior in scenario A is explained by simple explanation B"
"Technique C can fail in scenario D if conditions E and F hold"
One important claim, with sufficiently strong evidence, can be enough for a great paper! If you want multiple claims, I strongly recommend choosing claims that fit together in a cohesive theme - papers are far easier to understand, praise, share, etc if there is a coherent narrative, not just a grab-bag of unconnected ideas.

Depending on the strength of the evidence, you can adjust the confidence of a claim:

Existence-proof claims: "We found at least one example where X happens" (like the indirect object identification paper providing an existence-proof for self-repair)
Systematic claims: "X generally happens across a wide range of contexts" or "X is common"
Hedged claims: “There is compelling/suggestive/tentative evidence that X is true”
Narrow claims: “X is the best method for specific situations V & W, if your goal is objective Y”
Guarantees: “X is always true”[3]

Generally, stronger statements make for more interesting papers, but require higher standards of evidence - resist the temptation to overclaim for clicks!

When to Start?
Another thorny question is: When should you stop doing research and start writing up your research? This is a hard and subtle question that is, in many ways, a matter of research taste, but here is my general guide:

Write down a list of things you've learned
Review that list carefully, and ideally show it to someone else
Ask yourself how comfortable you would be defending the claim that you have provided meaningful, positive evidence for these results
Think about reasons why others might care about this
Focus on things you've done that have been hard or non-trivial and look for exciting elements
But generally, this is unfortunately just a hard thing to tell when starting out, and gets far easier with time and experience. If you can consult a more experienced researcher, definitely do.

A more meta piece of advice when starting out is to try to choose projects where the narrative will be pretty obvious, e.g. method X beats SOTA method Y in domain Z on metric W

Warning: Before moving into paper-writing mode, it's crucial to verify that your evidence is actually correct. An unfortunate fact is that many published papers are basically false or wildly misleading. Don't let this happen to you! Carefully check your critical experiments and, if possible, re-implement them through alternate pathways. Ideally, verify all experiments worth mentioning in the paper, or at least 75% of them.

Novelty
A common and confusing requirement for papers is that the results be novel, something that is not covered before. What exactly does this mean? Science is fundamentally about building a large body of knowledge. This means that your work exists in the context of what has come before. Novelty means it expands our knowledge.

The conventional definition of novelty can be annoying and, in my opinion, focuses too much on shininess and doesn't capture the more important aspect of whether our knowledge has expanded. Another way to put this is: Should I assign different probabilities to propositions I care about after observing the results of this paper?

Rigorous, at-scale replications of shaky results, negative results of seemingly promising hypotheses, and high-quality failed replications of popular papers are all very valuable contributions. I would personally consider these novel because they expand our knowledge. However, the revealed preferences of many reviewers and researchers suggest they do not feel the same way. Such is life.

I don’t want to go too far re criticising novelty: there are many cases where I am uninterested in a paper due to lack of novelty. This primarily occurs with methods that I expect to work when applied in standard settings, and I assign a high probability of success, so the project provides few bits of information. While knowing that such a method failed could be interesting, projects can also fail due to researcher incompetence or bad luck. Therefore, it is difficult to draw meaningful conclusions without evidence of researcher competence.

Leaving that aside, novelty can be hard to communicate. Given a paper on its own, it's difficult to tell what is and is not supposed to be novel:

Are the techniques used innovative or just standard techniques?
Does the claim represent a deep conceptual breakthrough?
Is it a very simple extension of standard ideas?
Is it a natural consequence of a more ambitious claim put forward in a different piece of work?
The main way to address this is to be extremely clear about what is and is not novel, especially in the introduction and related work, and to liberally cite the most relevant papers and explain why your work is and is not different.

How to find out what came before? Use a large language model. If you’re not already familiar with a relevant literature, LLMs are pretty great at doing quick literature reviews, e.g. Gemini Deep Research[4]. Reading the literature yourself is much better of course, but takes way, way longer and should have been done at the start project.

One reason this is very important is that, depending on what’s claimed as novel, the same paper could be perceived as either inappropriately arrogant or making a modest incremental contribution, depending on how the claims are presented.

Contextualizing your work within existing literature is particularly crucial for experienced researchers who are familiar with the field. Clear explanation in the introduction helps them quickly engage with your work and see what’s interesting, else it blurs into all other superficially similar papers they’ve read and doesn’t seem worth the effort.

There are a few problems with novelty as it is traditionally thought of

Novelty is often overemphasized - it incentivises going for ambitious but shaky claims over simple and rigorous insights.
This can mean that if there's an existing paper that provides a preliminary but shaky case for a claim, going and doing it properly can seem less exciting, even though this is in some ways a more useful scientific contribution, as it establishes a confident foundation for others to build upon.
Another complex question arises when you have legitimate complaints about prior work, and your work superficially looks derivative, but this is because you identified a significant methodological flaw or bug.
I recommend being clear that you have criticism, but it's important to remain professional while explaining what was flawed and why this matters, and how your work resolves it, without critiquing the authors or their motivations.
There are various social norms that are kind of annoying, such as citing being obliged to cite the first instance of a concept (even if later iterations are much clearer) and referencing a ton of vaguely relevant work even if it adds nothing to the paper - people can get offended if not cited. But this doesn’t detract from all the ways that citations genuinely strengthen a paper
I personally prefer to just do work that is optimised for scientific value, and shoe-horn it into a peer review friendly lens at the end, if applicable. But I’m in a fortunate position here, and there are real career incentives around getting published.

Two example papers of mine where being clear about novelty was tricky:

In my Othello work, I built directly on Kenneth Lee's paper that showed an Othello plane model had a world model found with non-linear probes. My contribution was demonstrating that it could be found with linear probes, which was interesting for a bunch of reasons to do with the linear representation hypothesis, but I needed to be careful to not claim credit for anything Kenneth did
In my refusal paper, our key result was that refusal is mediated by a single direction. But it was not novel to find that a concept was linearly represented, the significant part was doing it for refusal: a particularly interesting concept.
Other work had loosely tried to do this for refusal, but had less compelling results, so we had to explain why our’s was better (much larger effect sizes, more models, downstream tasks, etc)
We also showed that we could now jailbreak the model by removing this direction from the weights - the novelty was less that we could jailbreak models (that's already known to be easy with finetuning), but that we could do it with interpretability tools, and so cheaply, one of the first practical applications of interpretability (even if, you know, not quite for safety...)
Rigorous Supporting Evidence
A paper is worth little unless it can convince[5] the reader of its key claims. To do this, you need evidence. In machine learning, this typically means experiments. Below, I discuss how I think about what good experimental evidence looks like - see my research process sequence for more advice.

With claims the priority is being able to communicate the intuitions to everyone, but with experiments the priority is being able to justify it in full technical detail to an engaged, skeptical reader. You also want to explain what’s going on intuitively, to support your claims, but this is less key than actually having good, legitimate evidence.

A particularly important thing to get right is extensive red-teaming: you should spend a good amount of your time, both during the original research and now, red teaming your narrative. One of the main traps introduced by the framing of “find a great narrative” is the temptation to ignore inconvenient contradictory results - don’t let this happen to you. Tips:

Assume you've made a mistake - what is that mistake? Assume there's a hole in your case that your evidence supports your grand narrative - where is that hole? Try to break it apart.
Try to get other researchers, especially more experienced ones, to weigh in.
Make sure to extensively discuss limitations. If you notice issues, design and perform new experiments to test for them. This is all the more important the more ambitious or surprising your claims are.
When I read a paper with a bold claim, I have a strong prior that it is false, and I am constantly looking for holes. If I identify one, and the authors have not checked whether that is a real flaw, I will generally move on.
However, if they have preempted me and provide sufficient evidence that I can be confident it's not a flaw, then those papers can be incredibly exciting and insightful.
What does good evidence look like?

Good experiments distinguish between hypotheses: Often, you will have several plausible hypotheses for some phenomena. The point of an experiment is to have results that vary significantly depending on which is true (i.e. that provide Bayesian evidence) - if the results vary enough, and the experiment is sufficiently reliable, then one good experiment can falsify many hypotheses
Can you trust your results?:
How reliable is my experiment? Ask yourself: "How surprised would I be if it turned out to be complete bullshit due to a bug, error, noise, misunderstanding, etc.?" Investigate the most uncertain bits
How noisy is my experiment? If you ran similar experiments several times, how confident would you be that the results would be consistent? What is your sample size? What is your standard deviation? Are your results clearly distinguishable from noise? (There's a whole host of statistical theory here; your favourite LLM can probably effectively teach you about the basics.)
Statistical rigour: If you're doing some type of frequentist test[6], you probably shouldn't use p < .05 as a threshold. In stats heavy fields, like the social sciences, papers that report their central finding at .01 < p < .05, usually fail to replicate. If you're doing an exploratory approach, you should be skeptical of any result that isn't p < .001, as the number of possible hypotheses is vast.

Prior work discussing replicability is very strict on this point: "One prior study of 103 replication attempts [in psychology] indeed found a 74% replication rate for findings reported at p ≤ .005 and a 28% replication rate for findings at .005 < p < .05 (Gordon et al., 2021)". There are also various statistical reasons why true findings usually won't produce .01 < p < .05.[7]

This skepticism and sanity checking is especially key for particularly surprising or novel bits of evidence. Wherever possible, I will try to re-implement a key experiment from scratch or try to get at the same evidence via a somewhat different route, just to make sure that I'm not missing something crucial.
Ablation studies: When a paper introduces a complex new method, there are often several moving parts. For example, they may make changes A, B, and C to standard practice. If they then only evaluate the standard method or the method with all three changes, it's impossible to tell which changes are actually effective and necessary. It's good practice to remove one change at a time, observe its effect, and then repeat this process for each change.
Unknown Unknowns: How confident are you that there isn't some alternative explanation for your results that you're missing?
This is a gnarly one. You'll want to think hard about it, ideally ask other people for feedback and get their perspectives. However, ultimately, you may sometimes just need to move on after a reasonable effort and accept that you may have missed something.
Avoiding Misleading Evidence (Cherry-Picking and Post-Hoc Analysis):
Was this cherry-picked? Researchers can, accidentally or purposefully, produce evidence that looks more compelling than it actually is. One classic way is cherry-picking: presenting only the examples that look most compelling. This is particularly dangerous with qualitative evidence, like case studies.
While qualitative evidence can be extremely valuable, it’s important to note how cherry-picked it was. Ideally, provide randomly selected examples for context to give a fairer picture.
The main exception is if your claim is an existence proof. In this case, one example suffices, if it’s a trustworthy result.
Track pre/post-hoc analysis. It's important to clearly track which experimental results were obtained before versus after you formulated your claim. Post-hoc analysis (interpreting results after they're seen) is inherently less impressive than predictions confirmed by pre-specified experiments.
Be aware that even complex predictions suggested by a hypothesis can turn out to be correct for the wrong reasons
For example, in a toy model of universality, I came up with the key representation theory-based algorithm I thought the network would follow before we got our key pieces of empirical evidence. I felt very confident. However, follow-up work found that a different explanation, which also involved representation theory, was what was actually occurring.
Quality Over Quantity: Try to prioritise having at least one really compelling and hard to deny experiment, over a bunch of mediocre ones.
If you do have many experiments, often some are more compelling than others. Highlight the ones that most strongly support your claims in the main text and consider moving others to an appendix or referencing them more briefly.
Diverse Lines of Evidence Are Robust: On the flip side, it can be far better to have several qualitatively different lines of evidence all pointing to the same conclusion, rather than many very similar experiments that all use similar methodologies and standards of proof.
Qualitatively different basically means “given the result of experiment 1, how well can I predict the result of experiment 2?”
This can justify putting effort into weak lines of evidence; for example, qualitative analysis of some data points can be useful supporting evidence of a quantitative study, even if insufficient to carry a paper independently, as they make it less likely that the summary statistics hid a subtle flaw.
Distilling Experiments:
Often, at the end of a project, you'll have run many experiments, some of which felt around the edges of your core claims. But by this stage, you likely have a much clearer idea of what the most promising kinds of evidence are. If practical (considering time and resources), consider going back to run a more conclusive, decisive experiment using what you now know.
This could also involve scaling up: using more models, larger sample sizes, sweeping hyperparameters more thoroughly, running on more diverse datasets, etc.
Baselines are Crucial:
A common mistake is for people to try to show a technique works by demonstrating it gets "decent" results, rather than showing it achieves better results than plausible alternatives that people might have used or are standard in the field.
Implicitly you’re supporting the weak claim “method X works at all” not “method X is actually worth using in practice”
Sadly this is especially prevalent in fields like mechanistic interpretability, where the comparative need for qualitative evidence can lead to neglecting more rigorous and systematic quantitative comparisons against strong baselines - the best papers have both qualitative and quantitative evidence.
The subtlety of baselines: It's not enough to just have them; you must strive to have the strongest possible baselines. Put meaningful effort into making them good. Often, a "competitor" method can seem weak but can significantly improve with proper hyperparameter tuning, prompt engineering, or appropriate scaffolding.
There's a natural bias to invest more effort in making one's "cool, shiny" new technique look good than in optimizing "boring" baselines. Resist this. Rigorous comparison to strong baselines is critical for good science and for genuinely persuading informed readers.
The Guiding Question for Evidence:
Ultimately, the question to ask about your evidence is: "Should this update a reader's beliefs about my claims?" not "Does this fit the stereotypical picture of a rigorous academic paper?" While the latter often correlates with the former, your primary goal is genuine persuasion through sound evidence.
Eg if reading several dataset examples by hand is genuinely strong evidence of your claim, just report that and justify why it’s great evidence!
Reproducibility & Publishing code: Rigour can be in the eye of the beholder: if readers cannot understand or verify it for themselves, it’s far harder to consider it rigorous. So your paper will be made substantially more useful by providing more detail about your exact methods.
A particularly useful approach is sharing your code. This enables others to build on your work and clarifies any ambiguities left in the paper (there will always be some). More broadly, it provides transparency into your exact process.
If you have time, you should:
Ensure the codebase runs on a fresh machine
Write a helpful README that includes links to key resources like model weights or datasets (which can be easily hosted on Hugging Face)
Create a Python notebook demonstrating how to run the key components
Tragically, the world is complicated, and there is often no single clear recipe to deal with all edge cases in research. These considerations are guidelines to help navigate that complexity

Paper Structure Summary
How are these claims and experiments translated into a paper?

Abstract: Motivate the paper, present your narrative and the impact: explain your key claims and why you believe they are true - be as concise and high-level as possible, while still getting the point across. Give enough information for a reader to understand the key takeaways of your paper and whether to read further or not - they often won’t read further, so it’s key that they still get the gist!
The reader is coming in from a cold-start, and may have no idea what your paper is about - you need to help them orient fast, and indicate what “genre” your paper fits into
The rest of your paper exists to support the abstract
Introduction: Basically an extended abstract that fleshes out your narrative - explain your key claims, motivate them, contextualise them in the key parts of the existing literature. Explain your key experiments and the results and why this supports your claims. Ensure the reader leaves understanding the narrative of the paper, and whether to read further or not - they often won’t. This is basically a self-contained paper summary, don’t worry about “spoiling” the paper or repetition - with a complex idea, you want to repeat it in varied ways so that it sticks.
The introduction sets the structure of the rest of the paper
Main content: This is where the real technical detail lives. Clearly and precisely explain background concepts and results, your precise claims, what exactly you did for your experiments (in full detail, using appendices if need be, relevant baselines, etc), the results, what they mean and their implications, etc.
This should be tightly planned to support the key claims, not sprawling and comprehensive. For section and subsection you should have a clear answer for how it contributes to the narrative, and would be damaging to remove.
I recommend first planning out clear opening and closing sentences of each paragraph: what does the paragraph show and how does it fit into the paper?
Figures: Figures and tables are a key medium for communicating experimental results. Diagrams are great for communicating key ideas and claims. Put a lot of effort into your figures.
Good captions are also crucial - you need to given context on what the figure shows, the nuance and intended interpretation, and key technical detail. Ideally the reader will understand everything from just the figure and just the caption, though this is ambitious
Related work: This is a mini literature review - I generally put this after the main content and don’t think it’s super important. Giving context on a few key similar papers and how your work differs is crucial, but typically done in the intro.
Discussion (/limitations/conclusion/etc): A place to put all the high-level reflections - limitations of your work, future work, key implications, etc. This is not essential, but is nice if you have something worthwhile to say. Acknowledging key limitations is very important, and papers that don’t do this are substantially weaker and less useful (in my opinion).
Appendices: Everything else - in the main paper you need to care about being concise, but here you can do whatever you want. Often you want to briefly discuss something in the main paper and move all technical detail to the appendix. Appendices are pretty low stakes and rarely read except by superfans, so don’t stress them too much.
Analysing My Grokking Work
This is all pretty abstract. To concretise this, let’s look at my grokking paper through this lens. I’ve broken it down into claims, evidence, context and motivation, with commentary thrown in. This is somewhat stylised for pedagogical reasons, but hopefully useful!

Meta: This was a challenging paper to write!
There was significant technical detail to our claims and evidence, largely unfamiliar to readers - mech interp was very new, and we did something weird and novel. We needed to communicate our claims (the algorithm) and our experimental evidence, and justify why the evidence was believable, since there were no standard methods to follow
A good diagram was critical to explaining the algorithm:


Figure 1 from the paper, and lead image in the tweet thread

This was more like two papers - the reverse engineering, and the study of circuit formation, and we needed to compress both into the same page limit. Fortunately, they did fit a cohesive theme
Structure:

Claim: We fully reverse engineered a tiny transformer trained on modular addition
Meta: This is a general claim, but about a specific model
Context: We needed to explain the entire notion of reverse-engineering a model from its weights, as readers may not have been familiar. The motivation for why this is interesting is pretty obvious, but the goal is easy to misunderstand
Evidence: We show this with several lines of evidence: activation and weight analysis, and causal interventions
Claim: This circuit forms gradually, well before the point of sudden grokking -> grokking is a gradual from memorisation followed by removing memorisation
Context: The entire notion of grokking as covered in prior work
Meta: This is critical context to understand my paper, but is also prior work - I need to explain enough detail for unfamiliar readers to follow, but without excessive repetition, and cite it for them to see further details.
Motivation: Grokking is a big deal and surprised many people. We show it’s fairly different from what people think.
Evidence: We show this by designing convincing progress measures to track the circuit, and show that they shift well before grokking
The Writing Process: Compress then Iteratively Expand
Note: Check out my paper writing checklist for a concrete to-do list of what I recommend doing here

So, you have a list of claims and key experiments. Now, all you need to do is write the paper! I recommend an iterative process - start with a bullet point narrative, then a full bullet point outline, then flesh it out into prose, taking time to reflect at each stage. See above or the next section for more details on the actual structure of a paper, here I just try to convey the high-level strategy

A key challenge in paper writing is the illusion of transparency - you have spent months steeped in the context of this research project. You have tons of context, your reader does not. You’ll need to help them understand exactly what you are doing, in the large space of all possible ML papers, and address all the misconceptions and possible misunderstandings, even though to you it all feels obvious. This is a difficult skill - wherever possible, get extensive feedback from others to address

Spend far more time on early sections: Realistically, tons of people read the title of your paper, many read the abstract, some read the introduction/skim the figures, and occasionally they read the whole thing. This means you should spend about the same amount of time on each of: the abstract, the intro, the figures, and everything else[8]. (I’m only half joking)

Compress
You should start by compressing your work as much as possible. Some tips:

Verbally describe it to someone.
Bonus: Ask them what was most interesting, or to repeat it back to you
Plan out a talk.
Give your research notes to an LLM and ask it to summarize the key points.
After each, think about what's missing, what's extraneous, what’s inaccurate or misleading, and iterate.
This compression step is crucial because it forces you to identify:

The 1-3 concrete claims you believe to be true
Why these claims matter (brief motivation)
The crucial experimental evidence for each claim (ideally 1-3 key experiments per claim)
Next, critically evaluate this compressed version:

Do your experiments genuinely support your claims?
Are there ways your experimental evidence could be flawed or misinterpreted?
Could the evidence be true but the claim false? How?
As part of this process, write down common misconceptions, limitations, or ways someone might over-update on your work.

Iteratively Expand
Once you have a compressed list of bullet points that you are satisfied with, you should start iteratively expanding and developing them. After each step, stop, reflect, read through, and edit - rushing a step can lead to a lot of wasted time at the next step.

If you have a research supervisor/mentor, it is very valuable to get feedback at each stage - I find it way faster and easier to give feedback on a narrative or bullet point outline than being sent 8 pages of dense prose! Even if you don’t have a mentor, try to get feedback from someone[9].

Start with the compressed bullet point narrative - make sure you’re happy that this captures the narrative you want!
Write a bullet point outline of the introduction - the north star here is to communicate what your claims are, exactly, (including which parts are novel vs building on prior work), why they matter, and a high-level idea of why they are true
This involves more detail, key citations to the literature, more detailed motivations, etc. Generally it won’t get too technical, but can involve explaining a few crucial concepts.
Flow matters a lot here! Try to get feedback on how it feels to an unfamiliar reader, and how cohesive it feels
Write a bullet point outline of the full paper - covering the key experiments, results, methodology, background, limitations, etc
The north star is to convince a skeptical, engaged reader that your claims are true - give them enough information to understand your experiments and the results
A good outline is tight and minimal - every part of it should have a clear role in the overall narrative. If you don’t have a good answer to “what goes wrong if I cut this”, you should cut it.
Good figures are crucial to communicate results - plan these out, but leave making them to step 4
This can include writing the related work, or you can leave that to the end.
Results: Collect key experimental results and make first draft figures to show. Does this convincingly support your narrative? What’s missing? Which parts and complex and need more exposition, vs standard/unsurprising and can be sped through?
You can’t always get this done in advance, but it’s much better if you do - more time to refine, iterate, etc.
First draft: Flesh this out into prose and full technical detail
If you have writer’s block, try giving an LLM your outline, some relevant papers, and asking for a first draft. Do not just copy this into your paper, but I find that sometimes LLMs have good ideas, and that frustration with poor quality LLM write-ups can be a great way to break through writer’s block.
Edit it: Repeatedly pass over your first draft (and get feedback), clean it up, polish it, make the narrative as tight and clear as possible, cut out extraneous fluff, make the figures glorious, etc
This is worth spending a lot of time on, it can make a big difference!
The Anatomy of a Paper
OK, so what actually goes into a paper? What are the key components you’ll need to write, and what is the point of each?

Abstract
Check out the annotated abstract earlier for a concrete breakdown

An abstract should give a cold-start reader a sense of what the paper is about - what sub-field, what type of paper, what key motivating questions, etc. This is a key manifestation of the illusion of transparency:  you know exactly what your project is about but to your reader there is a large space of possibilities, and without any context may have completely incorrect priors and wildly misinterpret.

People will often leave your abstract then move on, unless strongly compelled - it’s a big deal to get right, and deserves high polish

A common approach is:

First sentence: Something uncontroversially true that clearly states which part of ML you're focused on (e.g., "Thinking models have recently become state-of-the-art across many reasoning tasks.")
Second sentence: Something that makes clear there's a need, something unknown, or a problem for your paper to solve (e.g., "The transition to reasoning models raises novel challenges for interpretability.") - this should convey (some of) the motivation
Now the reader is situated, you need to concisely communicate your claims. Again, illusion of transparency - they often won’t know your techniques, the work you’re building on, key ideas, etc. Abstracts should be as accessible as possible - use simple language as much as you can

Sentence 3: State the crucial contribution of this paper and why it is exciting - you’ll need to lose nuance, this is OK.
Optional: Sentence 4 should provide clarifying details on that claim, such as its meaning and how evidence could be provided if not obvious.
Include key definitions for any necessary jargon, though jargon should be avoided if possible, unless it’s standard in the field and useful to contextualise the paper within the field.
Each of the next few sentences should focus on either key experimental evidence or additional important claims. These can sometimes overlap, where a specific claim being true also supports the main claim.
Try to have 1 sentence per idea - this forces you to be concise, without getting overwhelming.
If possible, include a concrete metric or result in any of the above that gives readers a sense that your results are real and substantial.
This can look like folding in key evidence of a claim into the sentence introducing the claim.
Finally, close with motivation:

Final 1-2 sentences: Wrap up by reminding readers why the paper matters/is a big deal, its implications, and how it fits into the broader context.
This is also a good place to clearly state your standard of evidence, whether your work is:
A preliminary step towards…
Shows that method X should be used in practice
Shows that practitioners should take care when using method Y
Establishes best practices for Z
Provides compelling evidence that…
Introduction
The introduction is broadly similar to the abstract but more extended and in-depth. I proceed in roughly this order:

Paragraph 1: Context - What topic are we studying, what is the key motivating question, and why does it matter?
Optionally: 1 sentence on how our contribution answers it
It’s good to liberally cite papers here to establish things like ‘this is a real field’, ‘this problem matters’, ‘people are interested in it and have tried (and failed) to solve it/have solved variants’
Paragraph 2: Technical background - what do we know about this problem? What are the established techniques our paper rests on? Etc
It’s good to cite liberally here to establish that what you’re using are standard methods and concepts, and to give the reader more context.
Here and in paragraph 1 you want to better situate your problem in the broader strategic picture of the field. Why does this matter? What other work has been done here, and why is it inadequate?
Paragraph 3: Key contribution - What exactly is our main claim? Add key nuance, detail, context, etc.
Paragraph 3.5[10]: Our case - summarise the most critical evidence we provide that our main claim is true

[Optional]: More paragraphs for a second or third claim and the key case
Paragraph 4: Impact - What should you take away from this paper? What are the implications, why is it a big deal, who should take different actions as a result of the results, etc. This may be emphasising practical utility, pushing forwards basic science, correcting common misconceptions, etc.
Contributions: End with a bullet point list of concise descriptions of your key claims, ideally with concise descriptions of key evidence
You want something a reader can look at and decide if they’re impressed/interested
Note: Citing here isn't about performatively covering all the relevant papers[11]. It's about providing the context a reader needs to understand why your work is interesting and how it's limited. I try to have at least one citation for each step in an important argument, eg why

The introduction is where you have room to define key terms and concepts required to understand your claims, especially if they're somewhat technical.

It's often good to explicitly end with a bullet-point list of your contributions, which are basically just the concise claims you believe to be true, potentially with brief references to the supporting evidence.

Figures
Figures are incredibly important. Having clear graphs can be the difference between a very clear and easy-to-read paper and an incomprehensible mess.

To create a good figure:

Ask yourself, "What exactly is the information I would like someone to take away from this?" It's not just about finding the list of numbers output by your experiments and shoving them into some standard plotting software, you want to carefully choose a visualisation that emphasises the desired information and takeaway.
Ask yourself, "Why does this experiment tie back to my core claims? How would I like the reader to interpret these results? Which parts do I want to draw their attention to?"
Consider annotating a graph or, if there's one particularly important line, emphasising it
E.g. make it dark while the others are light and low opacity, or all the other ones of the same color.
Include standard elements like axis titles, a clear caption that explains what the figure is, how to interpret it, or at least where in the text they should look to understand what's going on.
Make sure the axis title and ticks are large enough to read, and have a good clear legend.
Often you can compress a fair amount of information into one graph - for example, using different sizes and shapes of markers on a scatterplot, different colors, etc.
For heatmaps, if your data is positive and starts at zero, use a color scale that is white at zero and dark at the max (in plotly, "blues" is good). If your data is positive and negative with zero as a meaningful neutral point, use a color scale where zero is white (in plotly, "RdBu" is good).
Avoid having reds and greens conveying key information, 4% of people are red-green colourblind
It can work well to combine several key graphs into one figure and make it your figure 1. E.g.:

Language models represent space and time:



Not all features are one-dimensionally linear:



Another kind of figure is an explanatory diagram rather than a graph. This can be a high-effort but very effective figure one, that gives people a sense of roughly what is happening in the paper. This should be something that would catch people's eye if you put it as the first image in a tweet thread about your paper. Some diagrams I liked (intentionally at several different levels of effortful):

Emergent Misalignment:



On the Biology of a Large Language Model:



CoT in the wild is not always faithful:



Refusal is mediated by a single direction:



My grokking modular addition work:



Main Body (Background, Methods and Results)
Most of the actual paper, by word count, should be about communicating your experiments and results in precise technical detail. To do good science, it is important that researchers can understand exactly what you did and what you observed, so they can draw their own conclusions rather than needing to take things on faith. For example, in interpretability, there are ways that a method can give completely useless answers if misapplied, so it’s crucial that I know if a paper did that, even though the detail might seem totally unimportant to the authors!

Ideally, you want to communicate the information at several different layers of abstraction. It's your job to ensure that readers understand:

The key background context required to disambiguate and understand your work - key terms, techniques, etc[12][13]

What your results are and how to interpret those results and their significance
What you actually did for your experiments
Why this was reasonable/well motivated/relevant to your claims
The specifics of various technical choices you made, and their implications for how to understand the results.
For structure, here’s a good default:

Background: to explain the relevant context and terms - in particular, please define terminology and crucial techniques!
If pressed for space, you can put a glossary of key terms/definitions as an appendix, I always appreciate this
If you’re defining something new for this paper, put this in a section which is clearly not about reviewing known things (a new section or separate subsection)
Methods: Explain the methods you used and why they are relevant to the problem
Results: Specify exactly how the methods are applied as experiments, and what the results are
If you have a bunch of experiments using fairly similar methods, put each in a different subsection
If the experiments for each claim are more boutique, or if there are several claims with different styles of evidence, then I try to give each type of evidence its own section while explaining how it ties back to the overarching theme, rather than a methods -> results section. People will forget about the first method before they see its results.

Discussion
Explaining the limitations of your work is a crucial part of scientific good practice. The goal of a paper is to contribute to our body of knowledge. Readers must understand the limitations of the evidence you provide to have a calibrated sense of what knowledge they have learned. And it's important that you put a good faith effort into documenting limitations because you know far more about your work than the readers, so they may miss things.

There is a common mistake of trying to make your work sound maximally exciting. Generally, the people whose opinions you most care about are competent researchers who can see through this kind of thing. And I generally have a much higher opinion of a piece of work if it clearly acknowledges its limitations up front. I’m not sure if this makes it easier or harder to get published.

This is also the place to discuss broader implications and general takeaways, future work you’d be excited about, reflections, etc.

Some people have conclusions too. Personally, I think conclusions are often kind of useless; the introduction should have explained this well. You can skip it

Related Work
Generally, related work is often treated like a bit of an annoyance and afterthought. The feeling that you need to cite lots of things that aren't actually relevant can be annoying, but sometimes there is very important work that has done similar things to you, and a reader might have seen that and wonder why your paper is interesting.

It's very important to clearly explain why what you did is different or, if what you did is not very different, either acknowledge this ("that was parallel work") or explain why your work is still slightly interesting in this context, or how you fixed a mistake in prior work (stated politely).

But that said, there’s a lot of annoying norms here and related works often add little value IMO - needing to cite a lot so you look like you’ve put in enough effort, covering minor or obscure things that aren’t particularly relevant, citing low quality works to be polite, making sure to cite the first instance of each thing, etc. Contextualising in the literature is important, but ideally I’ve already covered it in the introduction.

Related work is often put as the second section of the paper. Personally, I generally prefer it to be the penultimate section. I think related work should only be upfront if it plays an important role in motivating the paper - if your paper is very heavily tied to the surrounding literature, plugging a gap, correcting a mistake, or unlocking a new capability that would enhance various bits of prior work.

Appendices
Appendices are weird. They're basically the place you put everything that doesn't fit into the main paper. One way to think about it is that you're actually writing a much longer than nine-page paper - the main body and the appendices - but you've chosen a highlights reel for the first nine pages where you put all the absolutely key information. You place all the less crucial information in the appendices for readers to pick and choose from as they see fit.

In general, the crucial scarce resource you must manage is the reader's time and attention. The main body should be aggressively prioritized to make the most of this, be engaging, and communicate the most important pieces of information. But if you have a lot more to say than you can fit in there, then that's what appendices are for. A truly interested reader can go and take a look, though most won't.

Generally, appendices are held to a notably lower standard than the main body and will be read far less, so you should not feel obliged to put in meaningful effort polishing them. This is the standard solution to the dilemma when you want to include full technical detail but have done some fairly complex and convoluted work that just won't realistically fit.

Common Pitfalls and How to Avoid Them
Obsessing Over Publishability
Peer review is notoriously terrible for seeking truth. Reviewers often have biases, like favoring work that feels novel and shiny and exciting, or that doesn't feel weird or too new, or that doesn’t seriously challenge their existing beliefs. This has been shown in rigorous RCTs, where NeurIPS 2021 gave some papers two sets of reviewers and compared their decisions. The results… aren’t great:



I personally think that, at least in safety, doing good work that people respect matters more than getting into conferences, though both are nice. I’ve generally had fairly good results with just trying to write high-integrity work that explains why I believe it is interesting, and just trying to do good science and the work that I think is highest impact, even if it doesn't fit the academic mold.

But it’s pretty plausible to me that many of the people reading this are not in such a fortunate position, and that getting first author papers into top conferences would be a meaningful career boost, especially your first 1-2 papers. The strategy I generally recommend for my mentees is to spend most of the project doing the best scientific work they can. Then, as we approach the end of the project, we figure out how to wrap it up in a maximally conference-friendly package while writing and submitting it. If we did anything that made the work noticeably worse, we can undo it before uploading to Arxiv.

Unnecessary Complexity and Verbosity
Papers are seen as prestigious, formal, and highly intellectual artifacts. As a result, there's a tendency towards verbosity or trying to make things sound more complex and fancy than they actually are, so they feel impressive. I think this is a highly ineffective strategy. If I don’t understand a paper, I generally ignore it and move on, or assume it’s BS in the absence of strong evidence to the contrary. Often, the best papers just take some very simple techniques and apply them carefully and well. There’s a real elegance to being simple and effective.

People need to understand a paper in order to appreciate it and build on it and think it is interesting (except for superficial Twitter clickbait). Generally, you want to be precise, but within the constraint of being precise, be as simple and accessible as possible. Try to use plain language and minimize jargon except where the jargon is needed to precisely convey your meaning. You get points for quality technical insights, not for sounding fancy. Verbosity and overly complex language and jargon is actively detrimental to your paper’s prospects, IMO.

Not Prioritizing the Writing Process
People often do not prioritize writing. They treat it like an annoying afterthought and do all the fun bits like running experiments, and leave it to the last minute. This is a mistake. Again, your work only matters if people read and understand it. Writing quality majorly affects clarity and engagement. Writing is absolutely crucial and is a major multiplier on the impact of your work.

I typically recommend that people switch from understanding mode to distillation and paper writing a month before a conference deadline, if at all possible. You should want to spend a lot of your time iterating on a write-up, getting feedback, trying to make it clearer, thinking about weaknesses, etc.

Tacit Knowledge and Beyond
One irritation I have about the standard paper structure is that it heavily incentivizes being rigorous and maximally objective and defensible. Obviously, there are significant advantages to this, but I think that often a lot of the most valuable insights from a research project come in the form of tacit knowledge.

This might be:

This was hard and here are the steps we had to follow to get it to work.
Here are some ways we noticed our experiments catching fire and what we did to fix them.
Here's my fuzzy intuition of what's going on in the big picture - I can't fully defend it, but I'm reasonably confident this is true after several months of screwing around in this domain.
Here’s something I misunderstood for months before it suddenly clicked
Here’s a common misconception in this domain, or way people often misunderstand or overreact to our results
Here’s my advice to anyone replicating this work, especially how to find hyper-parameters and deal with the fiddly bits
Fleshing out a plan for future work directions you find particularly exciting.
I think this is really important, and I find it a real shame that this is often just discarded. I am personally a big fan of putting this kind of stuff as appendix A or as an accompanying blog post, where you can take as many liberties as you like.

Conclusion
Your research will only matter if people read it, understand it, engage with it, and ideally believe it. This means that good paper writing is a crucial skill, but often neglected.

The core process should be to find the concise claims you believe to be true, the strongest experimental evidence that you believe builds a robust case for these claims, and use this to craft a coherent narrative. Then flesh this out into a bullet point outline of the overall post, reflect on it, and ideally get feedback, and iteratively expand.

Again, this is a highly opinionated post about how I personally think about the process and philosophy of paper writing. I'm sure many researchers will strongly disagree with me on many important points, and the correct approach will vary significantly by field and norms.

^
 Note: I am in no way claiming that I follow this advice, especially in blog posts - this is my attempt to paint a platonic ideal, and advise on how to be a better person than I. Personally, I find actually writing academic papers pretty frustrating and much prefer blog posts

^
 I think writing great papers certainly helps, and if you’re new I recommend just trying to write the best paper you can, but there’s still a lot of depressingly perverse incentives from ML peer review

^
 I mention this purely for completeness. I have never seen a convincing guarantee in deep learning, neural networks are far too squishy

^
 OpenAI’s is also good, but Google’s is free!

^
 Well, at least “provide enough evidence to somewhat update their beliefs”, convince is a fairly high bar

^
 Which, admittedly, is fairly rare in ML as far as I’m aware

^
 Thanks to Paul Bogdan for these points on p-values

^
 Also the title, though I haven’t figured out how to productively spend 20% of my time on that yet…

^
 Paper swaps are a great way to get feedback - find someone else also working on a paper and offer to give each other feedback. Even if you have less time before the paper deadline, this tends to be a mutually beneficial trade.

^
 This can be its own paragraph, or part of paragraph 3

^
 Save that for the related work section…

^
 If something is super widespread knowledge, no need to cover it, but err towards defining things. E.g. I wouldn’t bother defining the transformer architecture or an LLM, but I would define a sparse autoencoder or steering vector

^
 If this is too long, you can move most of it to an appendix

 ---

 How to ML Paper - A brief Guide
Feel free to comment / share and happy paper writing! Also, please see caveats* below. 
If you like this, why not follow How to ML on Twitter and share the advice/love?
Canonical ML Paper Structure
Abstract (TL;DR of paper):
X: What are we trying to do and why is it relevant?
Y: Why is this hard? 
Z: How do we solve it (i.e. our contribution!)
1: How do we verify that we solved it:
1a) Experiments and results
1b) Theory 

Introduction (Longer version of the Abstract, i.e. of the entire paper):
X: What are we trying to do and why is it relevant?
Y: Why is this hard? 
Z: How do we solve it (i.e. our contribution!)
1: How do we verify that we solved it:
1a) Experiments and results, including comparison to prior SOTA if applicable
1b) Theory 
	2: New trend: specifically list your contributions as bullet points (credits to Brendan)
Extra space? Future work!
Extra points for having Figure 1 on the first page

Related Work:
Academic siblings of our work, i.e. alternative attempts in literature at trying to solve the same problem. 
Goal is to “Compare and contrast” - how does their approach differ in either assumptions or method? If their method is applicable to our Problem Setting I expect a comparison in the experimental section. If not, there needs to be a clear statement why a given method is not applicable. 
Note: Just describing what another paper is doing is not enough. We need to compare and contrast.

Background:
Academic Ancestors of our work, i.e. all concepts and prior work that are required for understanding our method. 
Usually includes a subsection, Problem Setting, which formally introduces the problem setting and notation (Formalism) for our method. Highlights any specific assumptions that are made that are unusual. 
Note: If our paper introduces a novel problem setting as part of its contributions, it’s best to have a separate Section.

[Problem Setting as separate section – only if the Problem Setting is novel, i.e. a contribution]

Method:
What we do. Why we do it. All described using the general Formalism introduced in the Problem Setting and building on top of the concepts / foundations introduced in Background.

Experimental Setup:
How do we test that our stuff works? Introduces a specific instantiation of the Problem Setting and specific implementation details of our Method for this Problem Setting. 

Results and Discussion:
Shows the results of running Method on our problem described in Experimental Setup. Compares to baselines mentioned in Related Work. Includes statistics and confidence intervals. Includes statements on hyperparameters and other potential issues of fairness. Includes ablation studies to show that specific parts of the method are relevant. Discusses limitations of the method. 

Conclusion:
We did it. This paper rocks and you are lucky to have read it (i.e. brief recap of the entire paper). Also, we’ll do all these other amazing things in the future. 
To keep going with the analogy, you can think of future work as (potential) academic offspring (credits to James).

Other Advice
Start with an outline rather than full text. Each line in the outline captures one idea and will correspond to one paragraph in the final version. It is much easier to change the outline of a building before building it. This is a great point in time to have conversations with others if you are unsure. 

Next, expand the outline, but keep the summary line as Latex comments ahead of every paragraph:

%TL;DR of paragraph
Lorem ipsum dolorem… (expansion of the TL;DR), a full paragraph of text that says the TL;DR as easily and clearly as possible. No fluff, not trying to impress, just saying what is and what is not and why think so. Simple, no? 

 This will a) keep you on track and b) make it easy for anyone providing feedback to quickly see what the overall flow is. 

I recommend writing the Abstract as soon as possible, even before results are finalised. Writing the Abstract first helps sharpen your focus and highlights issues with the paper / method / experiments. Of course you might have to make changes to the writing later, but that's much easier than trying to fix experiments / methods because you realise late that your story doesn't make sense.

A quick note on author ordering / inclusion, since this keeps coming up:
Who is an author? 
My rule of thumb is that anyone who worked on the project for a significant amount of time or contributed something that substantially improved the paper should be included.
In general, I recommend to err on the side of being generous with including authors. 
I also recommend to set clear expectations regarding authorship when discussing projects and ideas with potential collaborators (which is usually anyone). 
Ordering: 
First author – the person who did the bulk of the work, led the writing effort, often had the initial idea for the method.
Second author – typically substantial contributions. Usually awarded a * for equal contributions if they worked full time on the paper for a substantial period before the deadline.
Middle author(s) – broad spectrum! This could be somebody who contributed minor experimental results, someone who was part of the early conversations that led to the project idea or someone who provided usual feedback that materially improved the paper. It’s also a good place for collaborators who worked on the project, but whose results did not make it in the end. 
Second senior author – this is commonly the post-doc / PhD student that provides day-to-day guidance on the project. 
Senior author – usually the senior academic who is ultimately responsible for the project (since their name is on the line.. ;)  

Extremely common writing pitfalls and other advice (print this out and tick off?): 
Passive voice - unclear why, but this is a very common mistake. Passive gets way overused (e.g. here?). It’s clunky and obfuscates who did what. Avoid it if it can be avoided ;) (see more on this here – credit to Daphne Cornelisse). 
Be extremely clear on contributions. Never blur the lines between what had been done before and what you did. 
Be consistent with tense. Avoid switching at all costs, also avoid using future tense if it can be avoided: 
Eg. “In Section 3 we will show..”
Avoid filler words at all cost. Think about what you are trying to say and then say it, nothing else. Common filler words are “can”, “In order to”, “shall”  and many others. 
 Example: 
“The bank loan problem can be reformulated as a special subset of the contextual bandit problem” => 
“The bank loan problem is a special instance of a contextual bandit problem”
Once you have written the initial text, try to delete around ⅓ of the words. That’s typically how much “fluff” there is*.
Put yourself in the shoes of the reader. Is it clear why the current text is relevant and how it fits into the bigger picture? If not, explain. Guide them through the text.
Please use correct quotation marks in Latex ``correct quotation’’ (copy-paste this if unclear).  You can also use the function \enquote{correct quotation} from the \usepackage{csquotes} which automatically adjusts to the language you're using. 
Mathematical equations follow standard rules of orthography. Do not forget periods and commas after equations and avoid unnecessary colons. Example:
 ✔ The relationship between the radius and area of a circle is A = r^2 \pi.
 ❌ The relationship between the radius and area of a circle is: A = r^2 \pi.
Use “\citet” when authors are part of the sentence, e.g. “\citet{foerster2016learnnig} show ..”, and “~\citep” otherwise, e.g “.. recent work~\citep{foerster2016learning}”.
Use “\𝘶𝘴𝘦𝘱𝘢𝘤𝘬𝘢𝘨𝘦[𝗯𝗮𝗰𝗸𝗿𝗲𝗳=𝗽𝗮𝗴𝗲]{𝘩𝘺𝘱𝘦𝘳𝘳𝘦𝘧}” instead of simply “\𝘶𝘴𝘦𝘱𝘢𝘤𝘬𝘢𝘨𝘦{𝘩𝘺𝘱𝘦𝘳𝘳𝘦𝘧}” to make it easier for a reader to jump to the references and back. (credit to Boris)
Acronym + citation: “proximal policy optimisation \citep[PPO]{schulman2017ppo}..”.
Cite any claim that is not supported by your experiments and avoid grandiose language or overly broad claims - it makes it easy to attack the paper for no good reason.
Cite the correct version of published papers. Google scholar often defaults to the arxiv version, rather than the conference paper. Click “All 10 versions” and choose.
Check for broken references and citations in the PDF (indicated by ??).
Don’t leave writing the paper until the last minute. Aim for a complete draft a week before the deadline. 
Enable change-tracking in Overleaf and share directly with the email addresses of your collaborators. That way it’s in their UI. 
Introduce terminology that is specific to this work. If it risks being confusing, mention it right next to it "This should not be confused with...", and give an example.
Avoid synonyms for work-specific terminology at all costs.1 
Introduce any acronym before using it. The latex packages \usepackage[acronym]{glossaries} (see here) or \usepackage{acronym} can take care of this automatically. Avoid Rrandom Ccapitalisation (RC), even for method names and when introducing acronyms. Only proper nouns are capitalised, for example, Markov chain Monte Carlo (MCMC), partially observed Markov decision process (POMDP), etc.
The “Transformer” should be capitalised since it’s a proper noun and also changed the field ;) (credits to Albert).
Introduce only symbols and acronyms that you use in the paper. 
What is bold and what is italic? Up to you, but be consistent. I tend to use bold to highlight the key message in a paragraph and italic for key words in a sentence.
Avoid anthropomorphisms (“knowledge” etc) of AI algorithms.
Avoid subjective claims - usually adjectives are red flags.
“On the other hand” shouldn’t come without “On the one hand”.
Avoid repetition of words within a paragraph or sentence (“..that features that are..”).
Use simple language. Avoid rare words or sounding “fancy’’. For plenty of scientists (like myself) English is not the first language, don’t make life hard for them.
Footnotes should be after “.” and “,”.1
If a sentence gets too long and cumbersome, divide into two (credits to Berrin).
Avoid line breaks / paragraphs that result in lines containing a single (or few) words. It looks 
lazy.
Fill the page limit and avoid poor formatting / other whitespace. It looks lazy.

British or American English? Either is fine but be consistent. ​​Tip: on Overleaf switch "spell check" from "English" to "English (American)" or "English (British)" to make the distinction easier. 
Never copy-paste from other papers, unless you are verbatim quoting something. It’s much easier (and more ethical!) to write something from scratch than to try to modify something until it looks different. 
 Use the cleveref package for intelligent cross-referencing of figures, tables etc (credits to Sai).

Last not least - communicate plenty with all authors (i.e. at least daily for the last week) to stay on track and have fun!!

PS: Nothing here is binding but I think it makes it much easier for everyone if we stick to a basic structure when writing papers. Think of it like a broad convention that allows readers to quickly process papers. 

PPS: These broad best-practice suggestions are the result of writing papers with a fantastic set of mentors, students and other collaborators - all credits go to them! Special shout-out to my former PhD advisor, Shimon, who inspired many of the points here!

* Caveats:
I am sure there are many stylistic subtleties around writing that I am missing in this guide, but winning a literature or poetry competition is definitely not the goal. The goal is to write text that gets complicated content across efficiently. If in doubt just ask yourself if what you wrote is the simplest and clearest way to get the idea across. Also, you don’t need to follow every point religiously. But when you disregard one of these rules, you should be able to explain the motivation behind the rule and why your case is different. E.g., “Passive voice is normally a bad idea since it obscures the subject of the sentence. But in this specific sentence, passive voice was used because the object is the focus of the sentence. Also, the object was mentioned previously, so putting it first lessens the reader’s mental load.” (credits to Roger)