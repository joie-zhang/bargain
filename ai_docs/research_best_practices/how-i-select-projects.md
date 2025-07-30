How I select alignment research projects
by Ethan Perez, Henry Sleight, Mikita Balesni
10th Apr 2024
AI Alignment Forum
Youtube Video

Recently, I was interviewed by Henry Sleight and Mikita Balesni about how I select alignment research projects. Below is the slightly cleaned up transcript for the YouTube video.

Introductions
Henry Sleight: How about you two introduce yourselves?

Ethan Perez: I'm Ethan. I'm a researcher at Anthropic and do a lot of external collaborations with other people, via the Astra Fellowship and SERI MATS. Currently my team is working on adversarial robustness, and we recently did the sleeper agents paper. So, basically looking at we can use RLHF or adversarial training or current state-of-the-art alignment safety training techniques to train away bad behavior. And we found that in some cases, the answer is no: that they don't train away hidden goals or backdoor behavior and models. That was a lot of my focus in the past, six to twelve months. 

Mikita Balesni: Hey, I'm Mikita. I work at Apollo. I'm a researcher doing evals for scheming. So trying to look for whether models can plan to do something bad later. Right now, I'm in Constellation for a month where I'm trying to collaborate with others to come up with ideas for next projects and what Apollo should do. 

Henry Sleight: I'm Henry. I guess in theory I'm the glue between you two, but you also already know each other, so this is in some ways pointless. But I'm one of Ethan's Astra fellows working on adversarial robustness. Currently, our project is trying to come up with a good fine-tuning recipe for robustness. Currently working on API models for a sprint, then we'll move onto open models probably. 

How Ethan Selects Research Projects  
Henry Sleight: So I guess the topic for us to talk about today, that we've agreed on beforehand, is “how to select what research project you do?” What are the considerations, what does that process look like? And the rough remit of this conversation is that Ethan and Mikita presumably have good knowledge transfer to be doing, and I hope to make that go better. Great. Let's go. Mikita, where do you want to start?

Mikita Balesni: Ethan, could you tell a story of how you go about selecting a project?

Top-down vs Bottom-up Approach
Ethan Perez: In general, I think there's two modes for how I pick projects. So one would be thinking about a problem that I want to solve and then thinking about an approach that would make progress on the problem. So that's top down approach, and then there's a bottom up approach, which is [thinking]: “seems like this technique or this thing is working, or there's something interesting here.” And then following my nose on that. That's a bit results driven, where it seems like: I think a thing might work, I have some high-level idea of how it relates to the top-level motivation, but haven't super fleshed it out. But it seems like there's a lot of low-hanging fruit to pick. And then just pushing that and then maybe in parallel or after thinking through “what problem is this going to be useful for?”

Mikita Balesni: So at what point do you think about the theory of change for this?

Ethan Perez: For some projects it will be..., I think just through, during the project. I mean, often the iteration cycle is, within the course of a day or something. So it's not, it's not necessarily that if it ends up that the direction isn't that important, that it was a huge loss or sometimes it's a couple of hours or just a few messages to a model. Sometimes it's just helpful to have some empirical evidence to guide a conversation. If you're trying to pick someone's brain about what's the importance of this direction. You might think that it's difficult in some ways to evaluate whether models know they're being trained or tested, which is pretty relevant for various alignment things. You want to know, is the model just doing well on your training or testing distribution? Or how representative is the training or test performance going to be of the model's behavior in deployment or some other phase? And then I think that's the kind of thing where if you don't really have a specific idea, then it's a little bit hard to talk about it in the abstract. Are these situational awareness evals an important direction? A lot of the value of the project depends on the details. And so I think in that case, I've had an idea or woke up with an idea of, oh, I wonder if I ask this question, is the model going to respond in an interesting way? And then I just playground with that with the models. And then, oh, it seems like there's some interesting result here. Now I have a handle on how I would make some progress on this direction in evaluating situational awareness. That's an example of, a high-level example of how I've done that. And then okay, maybe I didn't spend half a day reading into the thinking that people have done on situational awareness or talking with people, but I broadly knew, okay, situational awareness is a thing that people care about. I think I can make progress on this. And so I just go for it. 

Henry Sleight: I would like to pick up on that because I'm wondering whether or not you're able to that kind of decision about experimenting empirically because you have good alignment intuitions. And maybe this isn't advice that would generalize to other people. 

Empirical Feedback and Prototyping
Ethan Perez: I think that's fair. I don't think I have particularly strong or great alignment intuitions. I think I know broadly a bunch of the areas that people think are relevant. I am less opinionated about them than other people. I think I'm often in the position where I'm like: situational awareness seems probably important. This is a thing that people talk about. Probably some work here would be useful. Generally, lots of the areas in alignment research are empirically bottlenecked in the sense of lots of people have been talking about a particular area for a while and haven't turned it into an experiment. Maybe because there's some translation that needs to be done from the idea to the experiment. If it seems like you have traction on some general direction that people are talking about, then this is one approach you could follow. If you're doing something that's a new direction, then it's certainly much harder. Maybe if you were doing activation steering for the first time, that's when you'd want to think a little bit more about what problem this is, tackling that kind of thing. I mean, even in the activation steering case, I think you could be, well, guiding the model behavior seems generally important. That's kind of what RLHF is doing. This seems RLHF like, and then do a quick experiment to derisk it and then talk with more people after that or brainstorm. The other thing that can help is just being in a research environment where there are a bunch of other people who have reasonable intuitions, because you'll just be drawn more to the interesting problems. So, I think that's the kind of thing that's helped me a bunch from being at Anthropic is that people are naturally drawn to certain kinds of problems, but you can have that at lots of other places, too. 

Mikita Balesni: I think that makes sense to me. It sounds like a lot of this is based on either, as you said, intuitions about what projects is good, what projects are good, or a sense of what people talk about. Once you have this general project idea of I want to do activation steering thing, or I have some interesting prompts for situational awareness evals, how do you assess whether you want to commit to this sort of thing?

Sharing Ideas and Getting Feedback
Ethan Perez: I'll just write up, here were my one day's worth of experiments. Here's what I think the motivation is. Here's why I think it's relevant. Then I'll share a doc with other people at Anthropic or outside of Anthropic and then ask for people's feedback. And then based on some of the specific comments people make and the general reaction, I'll go from there in deciding whether to continue. That's one mode for this bottom up approach. 

Henry Sleight: On that, what's the percentage of top down versus bottom up that you have noticed yourself doing, maybe over the last six months?

Pivoting Projects Based on Promising Results
Ethan Perez: It's a little bit hard to quantify because some projects are a bit of both. Where there's a top down reason - "we want to do adversarial robustness", for example. But then there's not necessarily a top down reason for the specific approach, whether or not to train the language model to be adversarialy robust, or whether to just have a test time monitor, or whether to use model internals. Those are things that we a bit empirical about - which approach we should take. We take a bottom up approach a bit of saying, "oh, it seems like this approach seems like it's working in parallel. We're doing some thinking through it, but we don't have a clear first principles reason why that's definitely the best way to go." But after we do some amount of that free form exploration, then we do sometimes switch to: "Oh, it does seem like, given these empirical results, here are a bunch of clear reasons why we should switch to this other approach." So I think it often switches back and forth. It's a bit hard to give a number, but it's not clear to me which one we're doing more of. 

Mikita Balesni: It sounds like I would probably just put this in a bucket of top down because there's the topic and the methods and here like you're saying that the methods are kind of bottom up, but the topic is top down. And is it roughly always the mix is in that shape that the topic is selected top down?

Top-down Approach and Low-Hanging Fruit
Ethan Perez: I mean, I think that the stuff with the situational awareness evals mini project that I've done, I guess I'd read a little bit about why people were interested in the situational awareness, but I wasn't like, this is the number one thing on my list of areas where we need alignment progress. I just hadn't explicitly ranked it. I was like, seems like I can get some interesting evidence on this just from playing around with the models. Maybe I can do a little bit more interesting investigation beyond that and then wrote it out. I think another case, I think it depends a little bit on what the research community needs as well. Sometimes there's just a bunch of low hanging fruit, for example: I think around the time of GPT-3, there was just tons of unexplored stuff on language models, and not many alignment researchers were working on language models. So there was just tons of opportunistic stuff you could do. We had this paper on learning from language feedback. I think that was just, you can think of that as an extension of RLHF, but getting a really important modality of human feedback in to be able to incorporate with language models. And that's an example of a thing where initially, I just tried some initial experiments there in a day or two to get some signal on, it does seem like language models can, in context learn from language feedback. That should motivate this other algorithm that builds off of that. And so if you're in that mode where there's just tons of low hanging fruit like that, just being opportunistic can get you a lot farther in a day than trying to think through the first principle stuff. But if you're in a regime where more work has been done or a lot of that's been picked up, that's the thing that can buy us away from the opportunistic stuff and the other thing that can buy us away concretely, right now, there's just a bunch of safety problems that do need to be solved now before we continue scaling up. I mean, that's the idea behind the responsible scaling plans. And so it's starting to get to the point where if we don't solve adversarially robustness, bad things will actually happen in the world. And so that does actually mean that we do need to sequence problems in a particular way depending on which problems are going to come up in what order. I think that's one of the reasons I'm excited about adversarial robustness now, because it's just, man, pretty soon models are going to be able to be jailbroken and it's actually going to be serious bad stuff's going to happen. And that's also relevant to longer term safety risk as well. So that's the thing that can drive more towards the top down approach. 

Mikita Balesni: So it sounds like in the early days of the field, when things are chill, you just can do a bunch of bottom up, but now things are shifting, and depending on your timelines, you probably want to. Or where you work, you want to choose different projects. 

Ethan Perez: I would say one specific additional thing is that each time there's a new model capability or there's a new model generation which is clearly more capable than previous model generation, that's definitely a time to be opportunistic because there's a ton of low hanging fruit that's not been explored when multimodal capabilities came online or code interpreter or GPT-4, that kind of thing. I definitely encourage all of my collaborators to play around with any of the new capabilities that are there because there's often just tons of low hanging fruit to pick at that point. 

Lessons Learned when Choosing Research Projects
Henry Sleight: What were some of your earliest lessons that you learned about choosing projects and how have they, over time turned out to be wrong?

Reading Prior Work
Ethan Perez: I think that I've gone through phases around how important reading papers or prior work are. Initially, I would just read a bunch. I would spend a month thinking about what I should work on and reading a bunch of the prior work. And then after that, I fell out of favor on that. I was like, oh, a bunch of the work isn't actually that relevant to alignment, and I'm working on actually different enough stuff that not all the past stuff is that relevant. And then, I think more recently I've gotten to a more nuanced view where, oh, for adversarial robustness, actually, there's just tons of relevant stuff and there's specific things that people haven't worked on from the alignment perspective. But a lot of the stuff is actually quite relevant and very directly relevant. I think that's also true to some extent about backdoors and sleeper agents. There's some relevant stuff, I think basically now my view is that it's pretty topic dependent. And I would just think and do a little bit of quick reading before starting in a new area about how much actually relevant stuff has been done, and then be critical in your evaluation on that. And then if you're like, oh yeah, I spent a day looking at this, none of the stuff looks like it's actually that relevant, then, I would just be like, experiments are going to be much more useful, your own experiments. And then if you do find some interesting stuff, then I would definitely chase that because it's free knowledge that's going to be really useful for your stuff. 

Henry Sleight: Cool. So it was on priors you were reading papers. Why should I ignore the field? Actually, you were getting a bunch more signal from your own experiments. But as it turns out, within some subfields, robustness, academia or open source community are doing plenty of work you can borrow from, basically. Cool. I was kind of nervous halfway through that because I was like, oh man, it's the classic rationalist AI alignment research to take to not read papers. Just do your empirical work. So cool. Glad for the nuanced opinion to rear its head again. Mikita, where are you at in that swing of opinions?

Mikita Balesni: I think it probably, I'm guessing that the reason Ethan updated here is not just because of Ethan became a lot more senior and smart. That too. But also with time, there has been much more progress in alignment. That is empirical and there's a lot more work to build on. Now there's control. There are a bunch of evals for various tendencies and for various capabilities. Now when you start a project, there's just dozens of less wrong posts to read, a bunch of papers to read, a bunch of people to talk to. Do you feel the same?That becomes more relevant with time. 

Ethan Perez: I think that's true, yeah, definitely the case. 

Mikita Balesni: Okay, so the one that I really wanted to ask is the one about duplication of work and value of replication results. This is where you and a bunch of others want to start working on something and you're worried that you'll duplicate work. How much should you care about that at all?

Ethan Perez: This is other people on the project? 

Mikita Balesni: No, other people, probably at other orgs doing something similar. And you want to, should we coordinate with them so that we don't do the same thing or should we collaborate or should we just do independent things?

Duplication of Work and Collaborations
Ethan Perez: I think my quick take would be very good to sync up with what they're doing in terms of just having a quick 30 minutes chat to understand what their angle is. And then I would have the same view as on related work where you're just very critical about is this actually solving the same kind of problem that we're trying to solve? And I would say almost always it's not, or there's some different angle in some exceptions and hopefully if the field alignment grows a bunch, then that will be more the case. But I think often the problem is somewhat different. And so in that case you can continue. But if they're working on the exact same problem and they've already made some relevant progress, you should just team up or just switch to some other area. 

Mikita Balesni: Right. And when you are both starting the same project, then you're both drafting docs for how it should go. Should you try to steer each other to different directions? Should you try to agree on what's the right way to do it? Do you want diversity or do you want alignment here?

Deciding to Collaborate or Not
Ethan Perez: I mean, there. I would just think about whether you're excited to collaborate with them and then think about, do they have the skills that would be relevant and complementing yours or just relevant to the project? Do you enjoy getting on with them? If that's the case, I think collaboration seems great. 

Mikita Balesni: Right. So there's this another consideration of maybe you really like them and you want to work with them, but for some reason you can't work with them. And in that case, if, for example, you're different orgs, different time zones or something, in that case, would you just continue working on this project or would you steer away so that two teams don't work on it at the same time?

Ethan Perez: I mean, I probably in that case, just picking one group to push on it seems good. It'll be a tricky conversation probably, but that's what I've done in the past. Or my mindset is somewhat, oh, it seems great that this other group is going to pursue this great idea. That means I don't need to do it. That means I can think about other stuff that I need to do. So I'm often just like, wow, it's exciting, I can go do something else.  

Henry Sleight: I'm curious, when you're selecting projects, how considerations about collaborators fit in. How bad is the worst project you'd work on for a really excellent collaborator or, vice versa? Would you never tolerate a collaborator under a certain bar, but would work on some projects if you could work with some of the best people you know?

Ethan Perez: I think if you are uncertain about some direction, but you would get to learn from someone who's pretty experienced or knowledgeable in some area, that seems pretty good. I've definitely seen people work with AI safety researchers who have controversial, strong takes, Dan Hendrickys, and then they just learned so much from Dan because they just got to build a mental model of, oh, wow. He said this thing that was really interesting and I can tell that when I talk with them next, I'm like, wow, they've learned so much. So I'm highly supportive of that. And then that's also a way to help you understand one worldview and then switch to Europe. People have worked with me and then disagreed with my ways of doing research tragically, and then gone off and realized, oh, I actually am interested in this direction. So I think it is actually just quite helpful for forming your own takes and opinions. 

Mikita Balesni: Do you have any project selection grooves that you go through that you would not recommend to others that are just the ways that you do this? But you wouldn't stand by this. 

Advice for Junior Researchers
Ethan Perez: If you are a junior researchers, the pure bottom up approach is going to be probably less effective unless you have a mentor who gives you a good direction or can give you feedback on what your high level direction is and that it's well motivated. My style is just very empirical, so I think I often just encourage people to run an experiment over, just to run tons of experiments, and I think that's very useful if experiments are very quick to run and highly informative, which is, for example, that's a big change from the previous era of machine learning where people had explored tons of different architectures or whatever. I think now the amount that capabilities have progressed year over year has been way more rapid than the previous. I mean, it was still rapid in the post deep learning pre LLM era, but I think it's much more rapid now. So I think this heavy experimental style is very useful. If you're working doing stuff that would be sampling on top of the OpenAI API, I think it's less relevant for potentially other areas. I think there depends a little bit on how much of the low hanging fruit has been picked in your area. If I had collaborators work on image related adversarial robustness, I'd be like, man, you probably do need to just spend more time reading some of the past stuff. And then I guess the other area where this is less relevant is for more conceptually heavy alignment research. For example, AI control, or potentially model organisms research, where it's just helpful to get more clarity on what is the exact thing that we want to demonstrate for making a model organism of some new failure. And you almost can't get empirical feedback on that in the sense of the uncertainty isn't empirical, it's just what would actually be interesting from a scientific standpoint. What are the research questions we want to answer? Or potentially, what are the things that would move the needle on policy? What are the demos that would be interesting to policymakers? So there, the feedback would be more, maybe for the policy thing, you would, talk with people who are very in touch with policymakers, or for the conceptual alignment, Evan Hubinger on the project. Just at some point just took a week and spent some time thinking about what kind of model organisms we should be demonstrating. And that's partly how we got to the sleeper agents particular model organism, as opposed to other things that I was. That's. I think those are a couple of cases where being less full on experimental is more the thing you should do. 

Mikita Balesni: The way you've talked about the research selection so far is kind of iterative, but have you ever had a moment of, aha, that completely changed the trajectory?

Pivoting projects
Ethan Perez: I think it's still pretty iterative when I have moments that change the trajectory of a project. So, for example, I was making evaluations for testing whether models will state that they're power seeking or state that they, whatever, have some other undesirable alignment property. Just to get some rough sense about what models were doing in that space. And then this was just going somewhat slowly, as in the turnaround time would be a week, and then you're writing things. So this wasn't by hand, I would write some by hand. And then I would send them to Surge, and then Surge would write a bunch of examples, and they're still very good at doing that. But it was just, not at this pace. The project wasn't going at the pace that I wanted to. And then you would meet some back and forth with the crowd workers being, oh, I realized this is a harder task to write evals for than I expected. And then at some point, I can't remember what happened, but I just had this strong sense. I was like, man, I feel like LLMs should really just be able to do this. I think this project should just be so much easier and so much faster. And then I remember on Friday, I told a bunch of my collaborators, I was like, I'm going to get this weekend. I'm going to get models to write the evaluations for me. And then the next day, I was like, okay, I was starting to sample from models and just being, can I make the same eval but with a language model?  And then I think basically by the end of the weekend, I had a bunch of working evals. And then I was like, oh, great, this is, so much easier. This is hundreds of evals. 

*laughter*

It made the whole rest of the project so much easier. I think that was partly driven by just butting up against this problem and being like, man, this is really harder than it needs to be. That's one case. Maybe another case where I had this was, this is a project with Geoffrey Irving when I was interning at DeepMind, where basically we were thinking about, oh, we want some set of hard questions that we can evaluate models on where we don't know the answers. And this is a test bed for things like scalable oversight, where we want to be able to evaluate how are models doing on questions where people don't know the answers. And so it seems good to just make a data set of hard questions where we can run techniques like AI safety via debate or iterated amplification and see, have a set of questions where we can see, what do the answers look like? Does it look like we're doing a reasonable job on these questions? And so that was the initial project motivation. And so I basically was just looking for, okay, what are the sources of these questions that are out there? There's lots of queries from Google that people have released. And so I was looking through data sets like that, and then I was looking through the data sets and not really feeling that excited about the questions. They're asking, they're just, pretty factual. People don't ask, why does God exist? And stuff. I mean, some people do, but it wasn't, the main thing that was happening on the Google. Then I just started doing more and more complicated stuff, like filtering down these, really large data sets or, maybe considering going through pre-training data. And then I just was talking with another collaborator or another person at DeepMind, and they were just like, oh, that seems kind of difficult. Have you tried generating the questions with language models? And I was like, man, that seems like a great idea. And then right after the meeting, she sent me some generated questions, and I was like, man, these are awesome. These are so great. These are, a great set of really hard questions. You can just ask the language models and they'll give you the questions. Now it sounds so obvious, but I think at the time, that was maybe, like two years ago or maybe even three. So it was just a little bit less obvious, at least to me. And I was less LLM pilled than I am now. But, I think then that week, we just changed the entire project direction to be, generating questions. And then from there, we ended up pivoting to this paper of red teaming language models with language models, where one of the obvious use cases of this is to find adversarial vulnerabilities in models. But that was partly driven by the shift to generating questions.

Mikita Balesni: I imagine things like this could come up more as models get better, as you said, when there are, new capabilities. For example, code execution came out and just, oh, this is a problem that that could solve. Sounds like also for that, you need to be banging with your head against some problem that is harder without that to really appreciate the value. 

Ethan Perez: I mean, I think that you could go faster towards those solutions than I did, and probably now I have definitely the instinct where I see people doing a task, and I'm like, try to do this with the language model, please. But I think that once you realize, oh, this direction, the broad direction of do the same thing you're doing, but automate it with the language model. That high level frame could be applied to, lots of different areas of alignment. 

Mikita Balesni: Are there any things that are immediate red flags for projects?

Red Flags for Projects
Ethan Perez: I think if the experiment loop is going to take more than a day, that's pretty rough because then you can't run experiments overnight and get feedback the next day. Definitely. If it's over a week or the other thing, that's a red flag is if there's a fixed cost of a month to run the first experiment, that's pretty rough. Sometimes there will just be such a good idea that someone should really do and burn the fixed cost. Maybe RLHF was like that. The fixed cost is plausibly many FTE months of competent OpenAI alignment team employees or something. I think there's cases like that where you can do it, but it's definitely by default it's a red flag and suggests that you should be thinking about simpler ways to run the experiment that still get you similar signal then. 

Other red flags, I guess I often surface these just by writing a two page doc on here's the problem I'm trying to solve and the motivation. Here's a potential solution. Here are the experiments we would run and then I send that doc out to a bunch of people and then who have takes on AI safety that I respect or maybe familiar with different experimental details about how easy this would be to run or whatnot, and then that kind of thing. Then I'll just get a bunch of comments and then often at that stage of the project, the most relevant feedback would be even if this project were extremely successful, it wouldn't be that important. I think, for example, people at Redwood Research give really good feedback along this vein. So I've learned a bunch from chatting with them and shared a bunch of my project docs with them. I think both on the project importance aspect and is this going to be easy to do? Aspect is probably the two main places where flags come up. 

Mikita Balesni: I wanted to ask for open source stuff. Is there any way to reduce the fixed cost for future versions of you or for other people? Is there any infra that could be built up that you wish was there, for example, you find missing when you do outside collaborations?

Open Source Infrastructure Needs
Ethan Perez: Good RLHF infra, that's a known correct implementation that works really well. That would be pretty valuable. 

Mikita Balesni: There is TRLX. 

Ethan Perez: I'm not familiar with how good the current stuff is. It's possible there's some existing stuff out there that's very good. But I think it's very tricky to get a good PPO implementation or whatever algorithm implementation for LLMs. Then making it more efficient is also a big thing because it's quite slow and then also making it so that the hyperparameters work well by default and you don't need to do a bunch of tuning. That's also pretty helpful. So it's possible there's something out there. My current knowledge, based on vibes from my external collaborators, is that there's not anything that's a slam dunk here. But, it's possible that improving on stuff like TRLX seems good, and I'm excited that they're doing the kind of stuff they're doing. 

Mikita Balesni: What's the way to improve that? If you don't have a baseline that works, how do you know that it is the correct implementation?

Ethan Perez: This is hard. I think having a person who knows what they're doing and I don't know has done it before, that's helpful. I don't know if I have great tips here because I haven't implemented RLHF from scratch. 

Mikita Balesni: Anything outside of that, other infra?

Ethan Perez: Not really. In general, I think it just changes a lot. Like every six months. What kind of thing would be helpful generally? I think for a while it was harder to fine tune models, and then OpenAI had the fine tuning API, which made it extremely easy to just write a function call, and then you can fine tune. And also it even sets the hyperparameters in pretty reasonable ways. So that made things much easier for my external collaborators. I think probably stuff like Langchain was helpful and unblocking some of the people I worked with. 

Mikita Balesni: Right. A thing that I find missing sometimes is a playground for open source models, and there's a big reason why I don't play with them as much as with OpenAI. 

Henry Sleight: So when should you switch off a project?

Tracking Multiple Promising Research Directions and Switching 
Ethan Perez: I think one trap projects commonly fall into is that they're making some progress, but not fast enough. So they seem maybe promising, but there's no clear result that shows this is really working. And then that can often be a big time sink, because there's just a lot of interesting, or there's a lot of stuff that seems like it might work, but hasn't hit a strong point yet there. It's really helpful. I find it really helpful to think about what are some other promising projects I could be doing, and then having a clear sense in my mind of "here are what the alternatives are". Because I think in the middle of a project, it's easy to be thinking, oh, this is net useful. Which is true. You would make progress on your current project by continuing to work on it. But I think knowing what the counterfactual is relative to a different project is pretty helpful, and then also just makes it easier emotionally to switch where you're switching, not because you're like, oh, this doesn't seem like it's working out, but more because, oh, I'm more excited about this other direction. So, in the case of the model written evaluations project that I discussed earlier, I was asking surge to make these evaluations in the background, 20% of my time while I was focusing on another project. And that project had some results, but then I was like, man, I feel like this evaluation process should be much easier. Then when I did this prototyping over the weekend, it was so much clearer. Wow, this direction is way more promising. The other stuff I was doing, and then it wasn't even a question for me to continue the other project because I was like, well, I have these much more promising results. I should just continue to push on those. I think that's one case for how to know when to switch projects is if you have something else that's more promising. Do you have any follow up questions there?

Mikita Balesni: From there, it sounds like your situation wasn't to stop your only project and think about alternatives. It sounds like you were working on at least one more project at a time, which allowed you to have this visibility awareness. Is that something you'd recommend?

Ethan Perez: Basically, mostly for junior researchers wouldn't recommend splitting projects because it's very hard to even do one project well. And that basically requires very full time effort. I think that basically each project has a fixed cost. Fixed cost is larger if you're a new researcher, in the sense of you currently have less context in the whole field. And so if you're doing 20 hours a week on a project as a first time reseracher, that's closer to a net output of 0 and all the gains come super linearly after that point. Once you're doing 30 hours a week, then you have this net 10 hours a week that you're able to contribute to the project because you have all the states and you're going to all the meetings and you read all the papers, and then you're able to do the right things. And then once you're going 60 plus hours a week, then you have the whole code base in your mind. All your thoughts are cached. You've just have so much state from talking with the other collaborators. You've already run a bunch of relevant experiments that are easy to build off of, so you've just paid down the fixed cost, and then there's a bunch of these additional benefits. Maybe your experiments have been working so you're more motivated. You've been able to talk with other people and get feedback, and they've given you great ideas. You're basically able to leverage all the stuff you did in the fixed cost time to make all of your other work that week go even better. 

Henry Sleight: So how does that wrap around back to what junior researchers should be doing when it comes to project selection, especially with Mikita's question of, projects that might not be working or something. 

Visibility into Alternatives
Ethan Perez: You can at least talk with people about what other directions seem promising without actually having to run the experiments. One of the things that was harder, at least for me early on, is running experiments quickly because I was like, whatever, just picking up how to code and stuff like that. So there it was more costly for me to run the experiment, or maybe some of the experiments I had in mind were pretty costly to run the first one. And so they're just talking with other people and getting excited about some other direction, being like, oh, I can actually write two page version of this project. And it seems like it's interesting to other people. That's often the thing that doesn't take that much extra effort per week. And then once you're at that point, then you could potentially be, okay, I'm explicitly going to pause this other project and spend a week derisking this other idea or this new model came out, and it seems like it would make AI Saftey via debate much more easy to run experiments on. So I'm going to now do that or whatever. And then after that week, you can assess. Does it make sense to continue on this?

Mikita Balesni: Right. So it sounds like you just should be siloed and then should think about other projects to people. 

Henry Sleight: Cool. Okay, Ethan, if there's one thing that you want people to walk away with from this chat, I guess, what would you want it to be?

Final Takeaway
Ethan Perez: The bottom up research approach is probably underrated in a lot of alignment research, where you just try some things that seem somewhat plausible and then see how they pan out and then go from there. You should think about whether that feedback applies to you. But I think I've seen lots of junior alignment researchers spend, a month reading about alignment, and that's highly encouraged by the alignment community. There's tons of writing out there. A lot of it seems like you're making intellectual progress on this, and it's very critical of different directions. And so you can feel demotivated and, oh, this person had a great critique of this idea and so maybe it's not worth working on and so I need to spend a month or three months figuring out what to do. I think just the opposite attitude of oh this seems actually plausible. I can do this fine tuning run in an hour. If I sit down and do it in a colab then you are making progress on some problem and then in parallel you can figure out how should I update this direction to be also including these results but also from talking with people how can I update it to be more relevant and I think just if more people on the margin did that that would be great. 

Henry Sleight: Awesome. Thanks so much. 

Ethan Perez: Cool. 

Mikita Balesni: Awesome. 

Henry Sleight: Ethan Mikita, thanks so much for coming. It was really great to do this. 

Mikita Balesni: Thank you Henry. 

Henry Sleight: See you outside when we just go back to our work. 

Ethan Perez: Cool. Thanks for hosting. 

Mikita Balesni: Yeah thanks.


 


36
Ω 14
Mentioned in
65
An Opinionated Evals Reading List
29
ML Safety Research Advice - GabeM
How I select alignment research projects
30
Michaël Trazzi
22
johnswentworth
26
Ethan Perez
4
Michaël Trazzi
New Comment


4 comments, sorted by top scoring
No new comments since 08/23/2024
[
-
]
Michaël Trazzi
1y
Ω11
30
0
Claude Opus summary (emphasis mine):

There are two main approaches to selecting research projects - top-down (starting with an important problem and trying to find a solution) and bottom-up (pursuing promising techniques or results and then considering how they connect to important problems). Ethan uses a mix of both approaches depending on the context.
Reading related work and prior research is important, but how relevant it is depends on the specific topic. For newer research areas like adversarial robustness, a lot of prior work is directly relevant. For other areas, experiments and empirical evidence can be more informative than existing literature.
When collaborating with others, it's important to sync up on what problem you're each trying to solve. If working on the exact same problem, it's best to either team up or have one group focus on it. Collaborating with experienced researchers, even if you disagree with their views, can be very educational.
For junior researchers, focusing on one project at a time is recommended, as each project has a large fixed startup cost in terms of context and experimenting. Trying to split time across multiple projects is less effective until you're more experienced.
Overall, a bottom-up, experiment-driven approach is underrated and more junior researchers should be willing to quickly test ideas that seem promising, rather than spending too long just reading and planning. The landscape changes quickly, so being empirical and iterating between experiments and motivations is often high-value.
Reply
[
-
]
johnswentworth
1y
Ω8
22
4
Meta: this comment is decidedly negative feedback, so needs the standard disclaimers. I don't know Ethan well, but I don't harbor any particular ill-will towards him. This comment is negative feedback about Ethan's skill in choosing projects in particular, I do not think others should mimic him in that department, but that does not mean that I think he's a bad person/researcher in general. I leave the comment mainly for the benefit of people who are not Ethan, so for Ethan: I am sorry for being not-nice to you here.

When I read the title, my first thought was "man, Ethan Perez sure is not someone I'd point to as an examplar of choosing good projects".

On reading the relevant section of the post, it sounds like Ethan's project-selection method is basically "forward-chain from what seems quick and easy, and also pay attention to whatever other people talk about". Which indeed sounds like a recipe for very mediocre projects: it's the sort of thing you'd expect a priori to reliably produce publications and be talked about, but have basically-zero counterfactual impact. These are the sorts of projects where someone else would likely have done something similar regardless, and it's not likely to change how people are thinking about things or building things; it's just generally going to add marginal effort to the prevailing milieu, whatever that might be.

Reply
[
-
]
Ethan Perez
1y
Ω14
26
16
Yeah, some caveats I should've added in the interview:

Don't listen to my project selection advice if you don't like my research
The forward-chaining -style approach I'm advocating for is controversial among the alignment forum community (and less controversial in the ML/LLM research community and to some extent among LLM alignment groups)
Part of why I like this approach is that I (personally) think there are at least some somewhat promising agendas out there, that aren't getting executed on enough (or much at all), and it's doable to e.g. double the amount of good work happening on some agenda by executing quickly/well
If you don't think existing agendas are that promising (or think they have more work done on them than they deserve), then this is the wrong approach
The back-chaining approach I'm advocating for is pretty standard in the alignment community, I think most alignment forum community researchers would probably endorse it. I'm also excited about this approach to research as well, and have done some work in this way as well (e.g., sleepers agents and model organisms of misalignment)
I'm guessing part of the disagreement here is coming from disagreement on how much alignment progress is idea/agenda bottlenecked vs. execution bottlenecked. I really like Tim Dettmer's blog post on credit assignment in research, which has a good framework for thinking about when you'll have more counterfactual impact working on ideas vs. working on execution.