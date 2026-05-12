|     |     |           |     |                   | Self-Distilled | Reasoner: |          |     |        |     |     |     |
| --- | --- | --------- | --- | ----------------- | -------------- | --------- | -------- | --- | ------ | --- | --- | --- |
|     |     | On-Policy |     | Self-Distillation |                | for Large | Language |     | Models |     |     |     |
SiyanZhao†1 ZhihuiXie2 MengchenLiu3 JingHuang3 GuanPang3 FeiyuChen∗,‡3 AdityaGrover∗1
Abstract post-training typically relies on reinforcement learning
methodssuchasReinforcementLearningwithVerifiable
Knowledgedistillationimproveslargelanguage
6202 raM 02  ]GL.sc[  3v43781.1062:viXra Rewards(RLVR)(e.g.,GRPO(Shaoetal.,2024;Guoetal.,
| model     | (LLM)     | reasoning    |     | by compressing | the      |            |                                              |            |         |     |               |         |
| --------- | --------- | ------------ | --- | -------------- | -------- | ---------- | -------------------------------------------- | ---------- | ------- | --- | ------------- | ------- |
|           |           |              |     |                |          | 2025; Team | et                                           | al., 2025; | Rastogi | et  | al., 2025; Yu | et al., |
| knowledge |           | of a teacher | LLM | to train       | smaller  |            |                                              |            |         |     |               |         |
|           |           |              |     |                |          | 2025)),    | supervisedfine-tuning(SFT)onhigh-qualityrea- |            |         |     |               |         |
| LLMs.     | On-policy | distillation |     | advances       | this ap- |            |                                              |            |         |     |               |         |
soningdatasets(Guhaetal.,2025;Teametal.,2025;Xi-
proachbyhavingthestudentsampleitsowntra-
aomi,2026),orknowledgedistillation,whererecentwork
| jectories | while | a teacher | LLM | provides | dense |     |     |     |     |     |     |     |
| --------- | ----- | --------- | --- | -------- | ----- | --- | --- | --- | --- | --- | --- | --- |
hasshownthatdistillationfromadvancedteachermodels
token-levelsupervision,addressingthedistribu-
canoutperformRLinbothperformanceandtrainingeffi-
| tion | mismatch | between | training | and | inference |     |     |     |     |     |     |     |
| ---- | -------- | ------- | -------- | --- | --------- | --- | --- | --- | --- | --- | --- | --- |
ciency(Yangetal.,2025;Xiaomi,2026;Lu&Lab,2025).
| inoff-policydistillationmethods. |     |     |     | However,on- |     |     |     |     |     |     |     |     |
| -------------------------------- | --- | --- | --- | ----------- | --- | --- | --- | --- | --- | --- | --- | --- |
policy distillation typically requires a separate, Despite their respective successes, each approach has in-
| often | larger, | teacher | LLM | and does | not explic- |                     |     |      |         |      |                |     |
| ----- | ------- | ------- | --- | -------- | ----------- | ------------------- | --- | ---- | ------- | ---- | -------------- | --- |
|       |         |         |     |          |             | herent limitations. |     | RLVR | suffers | from | inefficiencies | in- |
itlyleverageground-truthsolutionsavailablein cluding: (1)samplingagroupofresponsesperpromptis
reasoning datasets. Inspired by the intuition computationallyexpensiveandcanintroducehighvariance
that a sufficiently capable LLM can rationalize inestimatingthetruevaluefunction; moreover, whenall
external privileged reasoning traces and teach samples are either correct or incorrect, the gradient sig-
| its weaker |     | self, we | introduce | On-Policy | Self- |              |     |         |       |         |             |         |
| ---------- | --- | -------- | --------- | --------- | ----- | ------------ | --- | ------- | ----- | ------- | ----------- | ------- |
|            |     |          |           |           |       | nal vanishes | (Yu | et al., | 2025; | Zhao et | al., 2025); | and (2) |
Distillation(OPSD),alearningalgorithmwhere the reward signal is sparse and uniformly applied across
a single LLM acts as both teacher and student alltokensinthegeneratedoutput,neglectingfine-grained
withdifferentcontexts. Theteacherpolicycon- token-levelfeedback. Supervisedfine-tuningsuffersfrom
ditions on privileged information (e.g., verified exposure bias and weaker generalization (Agarwal et al.,
reasoning traces) while the student policy sees 2024;Chuetal.,2025). Traditionalknowledgedistillation
only the question; training minimizes the per- providesdensetoken-levelsupervisionfromateachermodel
tokendivergencebetweenthesedistributionsover butreliesonoff-policydata(Hintonetal.,2015). Recent
| thestudent’sownrollouts. |     |     |     | Wedemonstratethe |     |     |     |     |     |     |     |     |
| ------------------------ | --- | --- | --- | ---------------- | --- | --- | --- | --- | --- | --- | --- | --- |
advancesinon-policydistillation—whereastudentmodel
efficacy of our method on multiple mathemati- samplesitsowntrajectorieswhileateacherpolicyprovides
calreasoningbenchmarks,achievingsuperiorto- densetoken-levelsupervision—havedemonstratedsuperior
kenefficiencycomparedtoreinforcementlearn- sampleefficiencybycombiningthedistributionalrealism
ing methods and better performance over off- ofon-policytrainingwithdensefeedback(Agarwaletal.,
| policydistillationmethods. |     |     |     | Coderepo: | https: |     |     |     |     |     |     |     |
| -------------------------- | --- | --- | --- | --------- | ------ | --- | --- | --- | --- | --- | --- | --- |
2024;Lu&Lab,2025).
//github.com/siyan-zhao/OPSD.
Whileon-policydistillationhasshownstrongperformance,
itreliesonadistinctteachermodeltosupervisethestudent.
GiventhatmodernLLMsalreadyexhibitstrongreasoning
1.Introduction
|     |     |     |     |     |     | capabilities, | we  | ask this | research | question: | can a | model |
| --- | --- | --- | --- | --- | --- | ------------- | --- | -------- | -------- | --------- | ----- | ----- |
Recent advances in large language models (LLMs) have effectivelyserveasitsownteacherthroughself-distillation?
demonstratedimpressivecapabilitiesinreasoningandin- Ourapproachisinspiredbyhumanlearning: aftersolvinga
struction following. Achieving these capabilities during problemincorrectly,astudentcanexaminethecorrectsolu-
tion,rationalizeitssteps,andidentifywheretheirreasoning
*Equaladvising,†WorkdoneatUCLAandduringSiyan’spart-
|                 |     |         |      |          |            | failed. PriorworkhasshownthatforLLMs,evaluationis |     |     |     |     |     |     |
| --------------- | --- | ------- | ---- | -------- | ---------- | ------------------------------------------------- | --- | --- | --- | --- | --- | --- |
|                 |     | Meta.,‡ |      |          | 1UCLA 2HKU |                                                   |     |     |     |     |     |     |
| time internship | at  | Work    | done | at Meta. |            |                                                   |     |     |     |     |     |     |
3MetaSuperintelligenceLabs. ofteneasierthangeneration(Sunetal.,2024;Naor,1996).
|     |     |     | Correspondenceto: |     | SiyanZhao |     |     |     |     |     |     |     |
| --- | --- | --- | ----------------- | --- | --------- | --- | --- | --- | --- | --- | --- | --- |
Wehypothesizethatrationalization—explainingagivencor-
<siyanz@g.ucla.edu>.
|     |     |     |     |     |     | rectanswer—issimilarlyeasierthangeneration. |     |     |     |     | Motivated |     |
| --- | --- | --- | --- | --- | --- | ------------------------------------------- | --- | --- | --- | --- | --------- | --- |
Preprint.
1

On-PolicySelf-DistillationforLargeLanguageModels
|     |     |     | Construct two |     | On-Policy Self-Distillation |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | ------------- | --- | --------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
prompts for the
same LLM
Learning Objective
|         |     |        |     |     | Student Prompt |      |     | Teacher Prompt |     |     |     |     |     |     |
| ------- | --- | ------ | --- | --- | -------------- | ---- | --- | -------------- | --- | --- | --- | --- | --- | --- |
| Dataset |     | Sample |     |     |                |      |     |                |     |     |     |     |     |     |
|         |     |        |     |     |                | only |     |                | and |     |     |     |     |     |
Per-Token Divergence
|     |     |     |     |     | Student Policy |     |     | Teacher Policy |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | -------------- | --- | --- | -------------- | --- | --- | --- | --- | --- | --- |
: Problem
: CoT + Answer
On-Policy
Evaluate with
|     |                       |     |     |     |     | Sample |     |     |     | privileged  |     | Gradient only flow through |     |     |
| --- | --------------------- | --- | --- | --- | --- | ------ | --- | --- | --- | ----------- | --- | -------------------------- | --- | --- |
|     | Large Language Model  |     |     |     |     |        |     |     |     | information |     | the student's logits       |     |     |
,y⋆)}N
Figure1.OverviewofOn-PolicySelf-Distillation(OPSD):GivenareasoningdatasetS ={(x i ,weinstantiatetwopolicies
|     |     |     |     |     |     |     |     |     | x,y⋆). |     | i   | i=1 |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ------ | --- | --- | --- | --- | --- |
from the same LLM: a student policy p S (· | x) and a teacher policy p T (· | The student generates an on-policy response
(·|x,y⋆,yˆ
yˆ∼p S (·|x). Bothpoliciesthenevaluatethistrajectorytoproducenext-tokendistributionsp S (·|x,yˆ <n )andp T <n )at
eachstepn.Thelearningobjectiveminimizestheper-tokendivergenceD(p T ∥p S )alongthestudent’srollout.Thedivergenceherecan
beforwardKL,reverseKLorJSD.Crucially,gradientsbackpropagateonlythroughthestudent’slogits,allowingthemodeltoself-distil.
bythis,weinstantiateboththeteacherandstudentpolicies wefindstylistictokenscandominatethetrainingsignal
| from a | single | LLM. The | teacher | policy is | provided | with |     | ofmathtokens. |     |     |     |     |     |     |
| ------ | ------ | -------- | ------- | --------- | -------- | ---- | --- | ------------- | --- | --- | --- | --- | --- | --- |
privilegedinformationy⋆,suchastheground-truthanswer • WeevaluateOPSDonthreecompetition-levelmathemat-
or a reference chain-of-thought, while the student policy ical reasoning tasks, demonstrating that it matches the
conditionsonlyontheproblemx. Concretely,theteacher performanceofGRPOwithsignificantlyimprovedtoken
policy p (· | x,y⋆) conditions on both the problem and efficiencyandoutperformsupervisedfine-tuning.
T
theprivilegedanswer,whereasthestudentpolicyp (·|x) • We analyze the impact of different divergence objec-
S
observesonlytheproblem. Wepreservetheon-policytrain- tives, the effect of student generation length, and stu-
ingparadigmbysamplingtrajectoriesyˆexclusivelyfrom dent–teachergenerationstyles.
thestudentpolicy,whichthenreceivesdense,token-level
supervisionfromtheprivilegedteacherpolicy.
2.Background
WethereforeproposeOn-PolicySelf-Distillation(OPSD),
a framework in which a single model plays both teacher 2.1.KnowledgeDistillationforAutoregressiveLarge
| andstudentroles. |     | Thestudentsamplesitsowntrajectories |     |     |     |     |     |     |     |     |     |     |     |     |
| ---------------- | --- | ----------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
LanguageModels
| yˆ∼ p (· | | x);wethencomputetheper-tokendivergence |     |     |     |     |     |                                                    |     |     |     |     |     |     |     |
| -------- | ---------------------------------------- | --- | --- | --- | --- | --- | -------------------------------------------------- | --- | --- | --- | --- | --- | --- | --- |
| S        |                                          |     |     |     |     |     | Knowledgedistillationtransfersknowledgefromalarger |     |     |     |     |     |     |     |
betweenthestudentandteacherdistributionsandminimize
|                                |     |     |     |                        |     |     | teacher |     | model    | to a smaller | student   | model    | by training | the     |
| ------------------------------ | --- | --- | --- | ---------------------- | --- | --- | ------- | --- | -------- | ------------ | --------- | -------- | ----------- | ------- |
| itoverthestudent’sownrollouts. |     |     |     | Thisformulation(i)uses |     |     |         |     |          |              |           |          |             |         |
|                                |     |     |     |                        |     |     | student |     | to mimic | the          | teacher’s | behavior | (Hinton     | et al., |
on-policysupervision(thestudent’sowntrajectories),(ii)
|          |       |           |           |       |          |         | 2015; | Kim | &   | Rush, 2016; | Sanh | et  | al., 2019). | The core |
| -------- | ----- | --------- | --------- | ----- | -------- | ------- | ----- | --- | --- | ----------- | ---- | --- | ----------- | -------- |
| provides | dense | per-token | feedback, | (iii) | exploits | ground- |       |     |     |             |      |     |             |          |
y⋆, insight is that the teacher’s soft probability distribution
| truth solutions |     | and (iv) | requires | no separate |     | teacher |      |         |          |        |             |     |           |        |
| --------------- | --- | -------- | -------- | ----------- | --- | ------- | ---- | ------- | -------- | ------ | ----------- | --- | --------- | ------ |
|                 |     |          |          |             |     |         | over | classes | contains | richer | information |     | than hard | labels |
model. Thelearningprocessiscapturedbytheloss
|     |     |     |     |     |     |     | alone,        | as  | it reveals | the                                    | teacher’s | learned | similarities | be- |
| --- | --- | --- | --- | --- | --- | --- | ------------- | --- | ---------- | -------------------------------------- | --------- | ------- | ------------ | --- |
|     |     |     |     |     |     |     | tweenclasses. |     |            | Forauto-regressivelanguagemodels,given |           |         |              |     |
|yˆ|
|      |       |          |            | (cid:88) |     |     | adatasetS                                    |     | ={(x,y⋆)}wherexdenotesaninputandy⋆is |     |     |     |     |         |
| ---- | ----- | -------- | ---------- | -------- | --- | --- | -------------------------------------------- | --- | ------------------------------------ | --- | --- | --- | --- | ------- |
| L    | (θ)=E |          | E          |          |     |     |                                              |     |                                      |     |     |     |     |         |
| OPSD |       | (x,y⋆)∼S | yˆ∼pS(·|x) |          |     |     |                                              |     |                                      |     |     |     |     |         |
|      |       |          |            |          |     |     | thecorrespondingreferenceoutput,bothteacherp |     |                                      |     |     |     |     | andstu- |
|      |       |          |            | n=1      |     |     |                                              |     |                                      |     |     |     |     | T       |
(cid:16) (cid:13) (cid:17) dentp definetoken-leveldistributionsovervocabularyV.
|     |     | (·|x,y⋆,yˆ |     | (cid:13)p            |     |         |     | S   |     |     |     |     |     |     |
| --- | --- | ---------- | --- | -------------------- | --- | ------- | --- | --- | --- | --- | --- | --- | --- | --- |
|     | D   | p T        | <n  | ) (cid:13) S (·|x,yˆ | <n  | ) . (1) |     |     |     |     |     |     |     |     |
Traditionalsuperviseddistillationminimizesadivergence
Dbetweenteacherandstudentdistributionsaveragedover
afixeddataset:
Insummary,ourcontributionsareasfollows:
|     |     |     |     |     |     |     |     | L                      |     | (θ)=E |         | [D(p | ∥p )(y|x)], | (2) |
| --- | --- | --- | --- | --- | --- | --- | --- | ---------------------- | --- | ----- | ------- | ---- | ----------- | --- |
|     |     |     |     |     |     |     |     | SupervisedDistillation |     |       | (x,y)∼S |      | T S         |     |
• WeintroduceOn-PolicySelf-Distillation(OPSD),anovel
|           |     |              |          |       |        |         | where |     |     | D(p | ∥p )(y|x) |     |     | =   |
| --------- | --- | ------------ | -------- | ----- | ------ | ------- | ----- | --- | --- | --- | --------- | --- | --- | --- |
| framework |     | that enables | a single | model | to act | as both |       |     |     | T   | S         |     |     |     |
(cid:80)|
teacherandstudent,leveragingground-truthanswersto 1 y | D(p (·|y ,x)∥p (·|y ,x)) measures
|     |     |     |     |     |     |     | |y  | | n = | 1   | T <n | S   | <n  |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ----- | --- | ---- | --- | --- | --- | --- |
providedensetoken-levelsupervisiononstudentrollouts. per-tokendiscrepancy. However,thisoff-policyapproach
• Weintroduceaper-tokenpointwiseKLclippingmecha- suffersfromdistributionmismatch: thestudentencounters
nismthatstabilizestrainingandimprovesperformanceas different partial sequences y during auto-regressive
<n
2

On-PolicySelf-DistillationforLargeLanguageModels
|     |                     |     |     | SFT/Off-Policy |              |     | GRPO | On-Policy    |     |                         | On-Policy |     |     |
| --- | ------------------- | --- | --- | -------------- | ------------ | --- | ---- | ------------ | --- | ----------------------- | --------- | --- | --- |
|     |                     |     |     |                | Distillation |     |      | Distillation |     | Self-Distillation(Ours) |           |     |     |
|     | On-PolicyData       |     |     |                | ✗            |     | ✓    |              | ✓   |                         | ✓         |     |     |
|     | DenseLearningSignal |     |     |                | ✓            |     | ✗    |              | ✓   |                         | ✓         |     |     |
|     |                     |     |     |                | ✓            |     | ✗    |              | ✓   |                         | ✓         |     |     |
LowSamplingCost
|     | NoExternalTeacher |     |     |     | ✓   |     | ✓   |     | ✗   |     | ✓   |     |     |
| --- | ----------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
Table1.Comparisonoftrainingmethodsforreasoningtasks.On-PolicySelf-Distillation(OPSD)combinestheadvantagesofon-policy
trainingwithdensefeedbackwithoutrequiringanexternalteachermodel.
generationatinferencethanthoseseenduringtrainingon at the sequence level. The GRPO objective incorporates
thefixeddataset,leadingtocompoundingerrors. On-policy aclippedsurrogatelosstomoderatepolicyupdates,along
distillation (Agarwal et al., 2024; Lu & Lab, 2025; Xu with a reverse KL penalty to prevent excessive deviation
etal.,2024a)addressesthisbytrainingthestudentonits fromareferencepolicy:
| own generated | sequences |     | yˆ ∼ p | S (·|x), | obtaining | dense |     |     |     |     |          |      |     |
| ------------- | --------- | --- | ------ | -------- | --------- | ----- | --- | --- | --- | --- | -------- | ---- | --- |
|               |           |     |        |          |           |       |     |     |     |     | (cid:34) | |oi| |     |
token-level feedback from the tea cher on these on-policy 1 G 1
|                       |       |     |            |     |         |            |     |        | (θ)=E  |                   |              | (cid:88) (cid:88) |     |
| --------------------- | ----- | --- | ---------- | --- | ------- | ---------- | --- | ------ | ------ | ----------------- | ------------ | ----------------- | --- |
| samples:              |       |     |            |     |         |            |     | L GRPO |        |                   | x∼S          |                   |     |
|                       |       |     |            |     |         |            |     |        |        |                   | G            | |o |              |     |
|                       |       |     |            |     |         |            |     |        |        | o1,...,oG∼πθ(·|x) |              | i=1 i n=1         |     |
|                       |       |     |            |     |         |            |     |        | min(ρn |                   | ,clip(ρn     |                   | (5) |
| L                     | (θ)=E |     | [E         |     | [D(p ∥p | )(yˆ|x)]]. |     |        |        | A                 | i ,1−ε,1+ε)A | i )               |     |
| On-PolicyDistillation |       | x∼S | yˆ∼pS(·|x) |     | T       | S          |     |        |        | i                 | i            |                   |     |
|                       |       |     |            |     |         | (3)        |     |        |        |                   |              | (cid:35)          |     |
This approach connects distillation to imitation learn- −βD KL [π θ (·|x)∥π ref (·|x)]
| ing (Ross | et al., 2011), | where | the | student | iteratively | im- |     |     |     |     |     |     |     |
| --------- | -------------- | ----- | --- | ------- | ----------- | --- | --- | --- | --- | --- | --- | --- | --- |
provesbylearningfromtheteacher’sguidanceonitsown
|     |     |     |     |     |     |     |     | whereρn | = π θ (o | n |x ,o < | n ) istheimportanceratio,π |     | is  |
| --- | --- | --- | --- | --- | --- | --- | --- | ------- | -------- | --------- | -------------------------- | --- | --- |
outputs,combiningtheon-policyrelevanceofreinforcement i i n i < n θold
|     |     |     |     |     |     |     |     |     | π θ | (o i |x ,o | i ) |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---------- | --- | --- | --- |
learningwiththedenserewardsignalofsupervisedlearn- the policy before old the update, and ε controls the clipping
| ing, thereby | mitigating | exposure |     | bias while | maintaining |     |     | range. |     |     |     |     |     |
| ------------ | ---------- | -------- | --- | ---------- | ----------- | --- | --- | ------ | --- | --- | --- | --- | --- |
computationalefficiency.
WhileRLVRmethodshavedemonstratedstrongempirical
|     |     |     |     |     |     |     |     | performance,theyfacetwokeylimitations: |     |     |     | (1)thereward |     |
| --- | --- | --- | --- | --- | --- | --- | --- | -------------------------------------- | --- | --- | --- | ------------ | --- |
2.2.ReinforcementLearningwithVerifiableRewards
|     |     |     |     |     |     |     |     | signal is | sparse, | providing | only sequence-level | feedback |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --------- | ------- | --------- | ------------------- | -------- | --- |
ratherthantoken-levelguidanceonwhereerrorsoccur,and
| Reinforcement | learning | with | verifiable |     | rewards | (RLVR) |     |     |     |     |     |     |     |
| ------------- | -------- | ---- | ---------- | --- | ------- | ------ | --- | --- | --- | --- | --- | --- | --- |
hasemergedasapopularapproachforpost-traininglarge (2) when all sampled responses receive identical rewards
|          |                      |     |     |            |        |         |     | (all correct | or all | incorrect), | the advantages | become | zero, |
| -------- | -------------------- | --- | --- | ---------- | ------ | ------- | --- | ------------ | ------ | ----------- | -------------- | ------ | ----- |
| language | models, particularly |     | on  | tasks with | easily | verifi- |     |              |        |             |                |        |       |
ableoutcomessuchasmathematicsandcoding, usingal- preventinganypolicyupdatedespitethecomputationalcost
| gorithmslikeProximalPolicyOptimization(PPO)(Schul- |     |     |     |     |     |     |     | ofsampling. |     |     |     |     |     |
| -------------------------------------------------- | --- | --- | --- | --- | --- | --- | --- | ----------- | --- | --- | --- | --- | --- |
manetal.,2017)andGroupRelativePolicyOptimization
| (GRPO)(Shaoetal.,2024). |           |          |         |     |      |           |     | 3.Methods |     |     |     |     |     |
| ----------------------- | --------- | -------- | ------- | --- | ---- | --------- | --- | --------- | --- | --- | --- | --- | --- |
| GRPO                    | trains by | sampling | a group |     | of G | responses |     |           |     |     |     |     |     |
3.1.LearningfromVerifiableReasoningDataset
| {o ,o ,...,o | }fromthecurrentpolicyπ |     |          |          | foreachprob- |     |     |             |     |         |                     |       |     |
| ------------ | ---------------------- | --- | -------- | -------- | ------------ | --- | --- | ----------- | --- | ------- | ------------------- | ----- | --- |
| 1 2          | G                      |     |          |          | θ            |     |     |             |     |         |                     |       |     |
| x.           |                        | o   |          |          |              | r   | ∈   | We consider | a   | dataset | of problem-solution | pairs | S = |
| lem          | Each response          | i   | receives | a binary | reward       | i   |     |             |     |         |                     |       |     |
|              |                        |     |          |          |              |     |     | ⋆)}N        |     |         |                     |       | ⋆   |
{0,1}indicatingcorrectness. Themethodthenassignsad- {(x i ,y , where each x i denotes a problem and y
|     |     |     |     |     |     |     |     | i   | i=1 |     |     |     | i   |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
vantages to all tokens k = 1,...,|o | within response o isthecorrespondingreferencesolution,whichmayinclude
|     |     |     |     | i   |     |     | i   |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
usingagroup-normalizedreward: chain-of-thoughtreasoning. Forbrevity,weomitthesample
indexianduse(x,y⋆)todenoteagenericsamplefromthe
r −mean({r }G ) dataset. Wecanexploitlearningsignalsfromthisdataset
|     |       | i   |     | j j=1 |     |     |     |     |     |     |     |     |     |
| --- | ----- | --- | --- | ----- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|     | A i = |     |     |       | .   | (4) |     |     |     |     |     |     |     |
std({r }G ) fromdifferentways: Standardsupervisedfine-tuning(SFT)
j j=1
onS canbeviewedasoff-policydistillation/imitationlearn-
Thisformulationcanbeunderstoodthroughthevaluefunc- ingusingexperttrajectories,butitsuffersfromdistribution
tion lens: mean({r }G ) serves as a G-sample Monte mismatchbetweentrainingandinference. Reinforcement
j j=1
CarloestimateofthevaluefunctionV(x),whilethesparse learningfromverifiablerewards(RLVR),suchasGRPO,
binaryrewardr i representsthe(undiscounted)state-action addressesthisbyoptimizingon-policysamplesandassign-
valueQ(x,o ). Critically,alltokenswithinaresponseshare ingbinaryrewardsbycomparinggeneratedanswersagainst
i
thesameadvantage,astherewardsignalisprovidedonly y⋆. However,RLVRiscomputationallyexpensiveandthe
3

On-PolicySelf-DistillationforLargeLanguageModels
StudentPrompt
|     | Problem: |     | Find | the derivative |     | of  | f(x)=3x2+2x−5 |     | at  | x=2 |     |     |     |
| --- | -------- | --- | ---- | -------------- | --- | --- | ------------- | --- | --- | --- | --- | --- | --- |
Answer:
TeacherPrompt
|     | Problem: |      | Find        | the derivative |      | of       | f(x)=3x2+2x−5 |      | at              | x=2 |     |     |     |
| --- | -------- | ---- | ----------- | -------------- | ---- | -------- | ------------- | ---- | --------------- | --- | --- | --- | --- |
|     | Here     | is   | a reference | solution:      |      |          |               |      |                 |     |     |     |     |
|     | First    | find | f′(x)=6x+2, |                | then | evaluate | at            | x=2: | f′(2)=6(2)+2=14 |     |     |     |     |
After understanding the reference solution, please try to solve this problem
|     | using | your | own | approach | below: |     |     |     |     |     |     |     |     |
| --- | ----- | ---- | --- | -------- | ------ | --- | --- | --- | --- | --- | --- | --- | --- |
Answer:
Figure2. Promptexampleforstudentandteacherpolicies. Bothpoliciessharethesameparametersθbutdifferinconditioning
Theteacherreceivestheground-truthsolutiony⋆asprivilegedinformationbeforegeneration.
| context. |     |     |     |     |     |     |     |     |     |     |     | Toensureanaturaltransition |     |
| -------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | -------------------------- | --- |
beforeevaluatingthestudent’srollout,theteacherispromptedtorationalizeandgenerateitsownsolution.Notethattheteacherwon’tbe
generatingtokens—rationalizationisdoneimplictlythroughoneforwardpass.
rewardsignalissparse,providingsamefeedbackacrossall varyingtheconditioningcontext. Theteacherpolicycon-
tokensregardlessofwhereerrorsoccur. Alternatively,one ditionsonprivilegedinformation—boththeproblemxand
cantrainaprocessrewardmodel(PRM)toprovidedense, thereferencesolutiony⋆:
token-levelfeedbackduringRL.However,acquiringlabels
|                                                      |     |     |     |     |     |           |     |     |         | p (·|x,y⋆) |          | ≜ p (·|x,y⋆).    |            |
| ---------------------------------------------------- | --- | --- | --- | --- | --- | --------- | --- | --- | ------- | ---------- | -------- | ---------------- | ---------- |
| forPRMtrainingisprohibitivelyexpensiveanddifficultto |     |     |     |     |     |           |     |     |         | T          |          | θ                |            |
| scale(Lightmanetal.,2023;Zhangetal.,2025).           |     |     |     |     |     | On-policy |     |     |         |            |          |                  |            |
|                                                      |     |     |     |     |     |           |     | The | student | policy     | observes | only the problem | statement, |
distillationworks(Agarwaletal.,2024;Xuetal.,2024a;Lu
matchingtheinference-timecondition:
&Lab,2025)addressdistributionshiftbytrainingonthe
student’sownsamples,butrequireaseparate,oftenlarger,
≜
|         |       |     |         |              |     |         |        |     |     | p   | S (·|x) | p θ (·|x). |     |
| ------- | ----- | --- | ------- | ------------ | --- | ------- | ------ | --- | --- | --- | ------- | ---------- | --- |
| teacher | model | to  | provide | supervision. | We  | instead | seek a |     |     |     |         |            |     |
trainingsignalthatisdense,on-policy,anddoesnotrequire
|           |     |                   |           |           |      |           |     | Bothpoliciessharethesameparametersθ |     |     |     |                         | butdifferonly |
| --------- | --- | ----------------- | --------- | --------- | ---- | --------- | --- | ----------------------------------- | --- | --- | --- | ----------------------- | ------------- |
| external  |     | teachers          | or reward | models.   | This | motivates | our |                                     |     |     |     |                         |               |
|           |     |                   |           |           |      |           |     | intheirconditioningcontext.         |     |     |     | Toencouragetheteacherto |               |
| On-Policy |     | Self-Distillation |           | approach. | We   | summarize | the |                                     |     |     |     |                         |               |
naturallyevaluatethestudent’sgeneration,weaddaprompt
differencesofthesemethodsinTable1.
askingtheteachertogenerateanewsolutionafterseeing
|     |     |     |     |     |     |     |     | thereferencesolutionasshowninFigure2. |     |     |     |     | However,the |
| --- | --- | --- | --- | --- | --- | --- | --- | ------------------------------------- | --- | --- | --- | --- | ----------- |
3.2.On-PolicySelf-Distillation
teacherdoesn’tgeneratetokens,itonlydoesrationalization
implicitlythroughprefilling.
| Motivation: |     | Learningbyunderstandingsolutions. |             |          |     |     | We       |                                  |     |     |     |               |     |
| ----------- | --- | --------------------------------- | ----------- | -------- | --- | --- | -------- | -------------------------------- | --- | --- | --- | ------------- | --- |
| propose     |     | a different                       | perspective | inspired | by  | how | students |                                  |     |     |     |               |     |
|             |     |                                   |             |          |     |     |          | On-policysamplingfromthestudent. |     |     |     | Givenaproblem |     |
learn: whenstrugglingwithaproblem,ratherthanextended
x,thestudentgeneratesanon-policyresponse
trial-and-error,astudentcanexaminethesolution,under-
standthereasoning,andinternalizetheapproach. Similarly, yˆ=(yˆ ,...,yˆ )∼p (·|x).
|     |     |     |     |     |     |     |     |     |     |     | 1   | |yˆ| S |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ------ | --- |
ifamodelhasaccesstothecorrectanswerorreasoningy⋆
andissufficientlycapable,itcanrationalizethereasoning Bothpoliciesthenevaluatethisstudent-generatedtrajectory.
stepsandteachitself—analogoustoastudentreviewinga Ateachpositionn,theyinducenext-tokendistributionsover
solutionandretracingwhyitworks.Thisintuitionmotivates y ∈V conditionedonthesamestudentprefix:
n
weexploittheground-truthsolutiony⋆di-
ourframework:
|x,y⋆,yˆ
rectlyasprivilegedinformationduringtraining,enablingthe p S (y n |x,yˆ <n ), p T (y n <n ),
modeltoserveasitsownteacherwithoutrequiringexternal
|                                    |     |     |     |     |     |     |     | whereyˆ |     | ≜(yˆ ,...,yˆ |     | ).  |     |
| ---------------------------------- | --- | --- | --- | --- | --- | --- | --- | ------- | --- | ------------ | --- | --- | --- |
| rewardmodelsorlargerteachermodels. |     |     |     |     |     |     |     |         | <n  | 1            | n−1 |     |     |
Teacherandstudentpolicies. Weinstantiatetwocondi- Training objective: Full-vocabulary logit distillation.
tional distributions from the same language model p by Weinstantiateafull-vocabularydivergenceobjectivethat
θ
4

On-PolicySelf-DistillationforLargeLanguageModels
Algorithm1On-PolicySelf-Distillation(OPSD)
Require: ReasoningdatasetS ={(x ,y⋆)}N ;languagemodelp ;divergenceD(e.g.,JSD )
i i i=1 θ β
1: Letp S (·|x)andp T (·|x,y⋆)bethesamemodelp θ underdifferentconditioning.
2: whilenotconvergeddo
3: SampleaminibatchB ⊂S
4: forall(x,y⋆)∈Bdo
5: Sampleon-policyresponseyˆ∼p S (·|x)
6: Computethetoken-wisedivergencealongthestudentrollout:
|yˆ|
ℓ(x,y⋆)←D (cid:0) p
T
∥p
S
(cid:1) (yˆ|x)=
|y
1
ˆ|
(cid:88) D (cid:0) p
T
(·|yˆ
<n
,x,y⋆) (cid:13) (cid:13)p
S
(·|yˆ
<n
,x) (cid:1)
n=1
7: CalculatelossL OPSD (θ)← |B 1 | (cid:80) (x,y⋆)∈B ℓ(x,y⋆)andupdateθ
matchestheteacherandstudentnext-tokendistributionsat training signal to be dominated by stylistic patterns. To
eachposition. Givenastudent-generatedsequenceyˆ,define addressthis,weapplypointwiseclippingtothevocabulary-
thetrajectory-averaged,token-wisedivergence leveldivergencecontributions. LetD (p ∥p )denotean
f T S
f-divergence. At each token position n and vocabulary
D (cid:0) p ∥p (cid:1) (yˆ|x)≜ 1 (cid:88) |yˆ| D (cid:18) p (·|x,y⋆,yˆ ) entryv,define:
T S |yˆ| T <n (cid:18) (cid:19)
n=1 (6) ℓ(f) =p (v |·)f p S (v |·) .
(cid:13) (cid:19) n,v T p T (v |·)
(cid:13)p
S
(·|x,yˆ
<n
) ,
Wecomputetheclippeddivergence:
where p (· | x,yˆ ) and p (· | x,y⋆,yˆ ) denote dis-
S <n T <n |yˆ|
t a r n ib y u d t i i s o t n ri s bu o t v io e n r t d h iv e e n rg e e x n t c t e ok m e e n as y u n re ∈ suc V h . as H t e h r e e, ge D ne c ra a l n iz b ed e D c (f lip )(p T ∥p S )= |y 1 ˆ| (cid:88)(cid:88) min(ℓ( n f ,v ),τ).
n=1v∈V
Jensen-Shannon divergence JSD , defined for a weight
β
β ∈[0,1]as:
Alternative objective: Sampled-token distillation
JSD (p ∥p )=βD (p ∥m)+(1−β)D (p ∥m) throughpolicygradient. Followingrecenton-policydis-
β T S KL T KL S
(7) tillation methods (Lu & Lab, 2025), we form a sampled-
wherem=βp +(1−β)p istheinterpolatedmixturedis- token reward signal (a reverse-KL signal on sampled ac-
T S
tribution. Thisfull-vocabularyformulationprovidesdense, tions)andoptimizewithpolicygradient. Foreachposition
token-levelfeedback: theteacher,informedbyy⋆,exposes ninasampledsequenceyˆ,definetheadvantageterm
thestudenttotheentiredistributionoverplausiblenextto-
A (x,yˆ)=logp (yˆ |x,y⋆,yˆ )−logp (yˆ |x,yˆ ),
kensandguidesittowardreasoningpathsthatleadtothe n T n <n S n <n
correctanswer.
andoptimizethepolicy-gradient-styleobjective
Weminimizetheexpecteddivergencebetweenteacherand
(cid:20) (cid:20) |yˆ|
studentoveron-policystudentsamples: 1 (cid:88)
L(θ)=−E E A (x,yˆ)
(x,y⋆)∼S yˆ∼pS(·|x) |yˆ| n
L(θ)=E (cid:2)E (cid:2) D (cid:0) p ∥p (cid:1) (yˆ|x) (cid:3)(cid:3) . n=1 (9)
(x,y⋆)∼S yˆ∼pS(·|x) T S (cid:21)(cid:21)
(8) ×logp (yˆ |x,yˆ ) .
S n <n
Gradientsarebackpropagatedonlythroughthestudentpol-
icyp ,whiletheteacherp actsasafixedfull-distribution
S T A (x,yˆ) is treated as a constant with respect to θ (i.e.,
targetconditionedonprivilegedinformation(x,y⋆). n
gradientsdonotflowthroughtheadvantage),sothatgra-
dients take the usual policy-gradient form A ∇ logp .
n θ S
Per-Token Pointwise Divergence Clipping. In our ex-
Comparedtothefull-vocabularydivergenceobjective,this
periments,weobservethattoken-leveldivergenceishighly
on-policy shaping objective operates only on sampled to-
skewedacrossvocabularyentries: asmallsubsetofstylistic
kens,usingtheteacher’slog-probabilitiestoprovidedense,
tokensexhibitsmuchhigherdivergencethanmathematically
trajectory-levelshapingsignalswithoutexplicitlymatching
meaningfultokens(seeTable5). Thisimbalancecausesthe
thefulldistributionateachstep.
5

On-PolicySelf-DistillationforLargeLanguageModels
58
56
54
52
50
48
0 25 50 75 100
Gradient Update Steps
)%(
ycaruccA
21@gvA
AIME24
46
44
42
40
38
36
34
0 25 50 75 100
Gradient Update Steps
)%(
ycaruccA
21@gvA
AIME25
32
30
28
26
24
22
0 25 50 75 100
Gradient Update Steps
)%(
ycaruccA
21@gvA
HMMT25
42
41
40
39
38
37
0 10 20
Tokens Generated (×106)
)%(
ycaruccA
21@gvA
Average
1.0
0.8
0.6
0.4
0.2
0.0
10 30 50 70 90
Gradient Update Steps
sehctaB
fo
noitcarF
Zero Reward Std Frac. of GRPO
GRPO OPSD
Figure3. TokenEfficiencyofOPSD.WecompareOPSDandGRPOonQwen3-1.7Bunderthesameeffectivetrainingbatchsize,
reportingAvg@12accuracywithtrainingstepsandtotaltokensgenerated.Generationiscappedat1024tokensforOPSDand16kfor
GRPO.Atthesamenumberoftrainingsteps,OPSDusessignificantlyfewertokensbutoutperformsGRPOonallbenchmarks.Despite
samplingmoretokens,GRPOonlyreceivesabinaryoutcomereward,andstagnatesduetorewarddiversitycollapse(rightmostplot):
morethanhalfofitsbatcheshavezerorewardstandarddeviationwithin100steps,yieldingnogradientsignal. OPSDsidestepsthis
disadvantageofoutcome-basedrewardsbylearningfromadensedistillationlossevenwithfewergeneratedtokens.
OPSDasdense-rewardpolicygradientandcomparison Baselines. Wecompareagainsttwomethodstrainedonthe
toSTaR. TheobjectiveinEquation(9)canbeseenaspol- samedataset: (1)SFT,standardsupervisedfine-tuningon
icygradientwithdense,token-levelrewards. InAppendix experttrajectories,whichcanbeseenasoff-policydistilla-
SectionD,weformalizethisandcontrastwithSTaR(Ze- tionfromamorepowerfulLLMthatgeneratedthereasoning
likmanetal.,2022),acloselyrelatedmethodthatalsouses traces;(2)GRPO(Shaoetal.,2024),grouprelativepolicy
thesamemodeltogeneratereasoningtraces,thenperforms optimizationwithbinaryoutcomerewardsverifiedagainst
rejectionsamplingfollowedbySFToncorrecttraces. This ground-truthanswers. Themaxgenerationlengthissetto
procedurecanbeviewedaspolicygradientwithasequence- 16k.
levelbinaryrewardthatassignsidenticalcredittoalltokens
Implementationdetails. Wefixtheteacherpolicytobe
andvanisheswhensamplesareincorrect.Incontrast,OPSD
theinitialpolicy,ratherthanthecurrentlyupdatinglearning
providesfeedbackateverytokenpositionregardlessoffinal-
policy,aswefindthishelpsstabilizetrainingandimplicitly
answercorrectness.
actsasregularizationtopreventexcessivedeviationfrom
theinitialpolicy. Weusefull-vocabularylogitdistillationin
4.Experiments ourexperiments. AllexperimentsareconductedonA100
orH100GPUswithLoRA(Huetal.,2022). Moreexperi-
Weconductcomprehensiveexperimentstoanswerthefol-
mentaldetailsareinAppendixB.
lowingresearchquestions:
(1) HowdoesOPSDcomparetoSFTandGRPOinrea- 4.2.MainResults
soningperformanceandsampleefficiency? (§4.2)
Table2reportsresultsoncompetition-levelmathematical
(2) Howdoesper-tokenpointwiseKLclippinginOPSD
reasoningbenchmarks.OPSDconsistentlyoutperformsSFT
helpstabilizingtraining? (§4.3.3)
andimprovesoverthebasemodelacrossallscales,match-
(3) Whatistheeffectofgenerationstyle,generationlength
ingorexceedingGRPOineverysetting. Notably,OPSD
onperformance? (§4.3.4)
achievesthesegainsusingonlyasinglerolloutperproblem
(4) Doesfull-vocabularylogitdistillationprovidebenefits
andconvergeswithin100steps,witheachproblemrequir-
oversampled-tokenpolicygradient? (§4.3.5)
ingonly1024sampledtokens,whereasGRPOrequires8
rolloutsof16ktokenseachandmayexhibitperformance
4.1.ExperimentalSetup
degradation in later steps due to entropy collapse—with
Models and datasets. We experiment with the most of reward standard deviations within a group being
Qwen3(Team,2025b)modelfamilyatthreescales:Qwen3- zerounderthisOpenThoughtsdataset,yieldingnolearning
1.7B,Qwen3-4B,andQwen3-8B,usingtheinstruct-tuned signalandwastingsamplingbudget. Wealsoobservecon-
versions. Fortrainingdata,weusethemathematicalreason- sistentperformancedegradationunderSFTacrosstasksand
ingsubsetofOpenThoughts(Guhaetal.,2025),sampling modelscaleswhentrainedonthesamedataset,whichwe
up to 30K problem-solution pairs with chain-of-thought attributetotheconcisereasoningstyleofthegroundtruth
reasoning. Weevaluateoncompetition-levelmathematics solutionswhichhasreducedreasoninglengthsattesttime.
benchmarksincludingAIME2024,AIME2025,HMMT WeattributeOPSD’stokenefficiencytodensetoken-level
2025. supervisionfromtheteacherdistribution,andwehypoth-
6

On-PolicySelf-DistillationforLargeLanguageModels
Table2.PerformancecomparisononmathematicalreasoningbenchmarksforQwen3models.WereportAvg@12underthesampling
configurationrecommendedintheQwen3blog(temperature1.0,maximumgenerationlength38k);fulldetailsareprovidedinTable8.
ForOPSD,weevaluatecheckpointsevery20stepsupto100stepsandreportthebestscore.ForGRPO,wereportthepeakperformance
within500trainingsteps,thoughwefindGRPOperformancetodecreaseforsometasksduetoentropycollapseinlatersteps.ForSFT,
wetrainonthesamenumberofsamplesasOPSD.SFTperformancedegradesduetofine-tuningonconcisereasoningsolutionsand
reducesgenerationlengthattesttime,whereasOPSDtransformsthemintodenselearningsignalthroughrationalization.
| Method | AIME24 | AIME25 | HMMT25 |     | Average |     |     |
| ------ | ------ | ------ | ------ | --- | ------- | --- | --- |
Qwen3-8B
| Base(Instruct) | 75.8 | 65.6 | 43.9 |     | 61.8 |     |     |
| -------------- | ---- | ---- | ---- | --- | ---- | --- | --- |
| +SFT           | 72.3 | 64.2 | 42.9 |     | 59.8 |     |     |
| +GRPO          | 76.4 | 68.9 | 46.7 |     | 64.0 |     |     |
| +OPSD          | 77.8 | 70.8 | 45.8 |     | 64.8 |     |     |
Qwen3-4B
| Base(Instruct) | 74.9 | 66.4 | 42.2 |     | 61.2 |     |     |
| -------------- | ---- | ---- | ---- | --- | ---- | --- | --- |
| +SFT           | 70.2 | 62.3 | 43.4 |     | 58.6 |     |     |
| +GRPO          | 75.6 | 68.1 | 44.4 |     | 62.7 |     |     |
| +OPSD          | 76.4 | 68.3 | 46.1 |     | 63.6 |     |     |
Qwen3-1.7B
| Base(Instruct) | 51.5 | 36.7 | 23.1 |     | 37.1 |     |     |
| -------------- | ---- | ---- | ---- | --- | ---- | --- | --- |
| +SFT           | 48.4 | 36.3 | 22.7 |     | 35.8 |     |     |
| +GRPO          | 51.1 | 38.3 | 23.7 |     | 37.7 |     |     |
| +OPSD          | 57.2 | 43.9 | 29.2 |     | 43.4 |     |     |
esizethatearliertokensmaycontributemoretoeffective scheme for stability. Forward KL consistently yields the
distillationastheycouldrepresentmorecriticalbranching strongestgains,improvingperformancefrom36.7to43.9
pointsinthereasoningprocess. at step 50 and remaining above the baseline at step 100.
|     |     | In contrast, | reverse | KL  | and JSD provide | limited | or nega- |
| --- | --- | ------------ | ------- | --- | --------------- | ------- | -------- |
Asshownin Figure3,OPSDachieveshighertokenlearn-
|     |     | tiveimprovements. |     | WethereforeadoptforwardKLinall |     |     |     |
| --- | --- | ----------------- | --- | ------------------------------ | --- | --- | --- |
ingefficiencywithin100stepsoftrainingascomparedto
remainingexperiments.
GRPO.Within100steps,GRPO’sperformancestagnates
withlesslearningsignalwhentheoutcomerewardwithinas
|     |     | Table3. | ComparisonofdivergenceobjectivesonAIME25with |     |     |     |     |
| --- | --- | ------- | -------------------------------------------- | --- | --- | --- | --- |
samplinggroupremainsthesame,leadingtozerogradient.
Qwen3-1.7B.WereportAvg@12atdifferenttrainingsteps.For-
TheseresultssuggestthatOPSDmayextractlearningsignal
wardKLsignificantlyimprovesperformanceoverthebasemodel,
fromthesamereasoningdatasetsmoreefficientlythanboth whilereverseKLandJSD(β = 0.5)showlimitedornegative
GRPOandSFT,whilesubstantiallyreducingtrainingtime. gains.
|     |     | Method |     |     | Base | Step50 | Step100 |
| --- | --- | ------ | --- | --- | ---- | ------ | ------- |
4.3.AblationStudies&Discussions
|     |     | ForwardKL(KL(p |     | ∥p  | )) 36.7 | 43.9 | 41.1 |
| --- | --- | -------------- | --- | --- | ------- | ---- | ---- |
T S
In this section, we conduct extensive ablations to study ReverseKL(KL(p ∥p )) 36.7 37.5 35.0
S T
keydesignchoicesinOPSD,including(1)thedivergence JSD(β =0.5) 36.7 36.9 39.0
objective,(2)thegenerationstylesofthestudentandteacher
(e.g.,thinking-modeon/off),(3)theeffectofper-tokenKL
clipping,(4)theimpactofstudentgenerationlength,and(5) 4.3.2.EFFECTOFGENERATIONSTYLESAND
comparisonbetweenfull-vocabularylogitdistillationwith PER-TOKENKLCLIPPING
sampled-tokendistillation.
AnotherkeydesignchoiceinOPSDisthegenerationstyle
|     |     | of the | student and | teacher | models, | as it determines | both |
| --- | --- | ------ | ----------- | ------- | ------- | ---------------- | ---- |
4.3.1.EFFECTOFDIVERGENCEOBJECTIVE
whichtokensthestudentlearnsfromandthestyleofsuper-
AkeydesignchoiceinOPSDisthedivergenceusedforper- visionprovidedbytheteacher. Qwen3modelssupporttwo
tokendistributionmatchingbetweentheprivilegedteacher generationmodes: ThinkingModeon(TM-on), inwhich
andthestudent. WecompareforwardKL,reverseKL,and themodelproducesself-reflectivechain-of-thoughttokens,
JSD on AIME25 with Qwen3-1.7B in Table 3. All ob- andThinkingModeoff (TM-off),inwhichitgeneratesre-
jectives are evaluated under the same pointwise clipping sponsesdirectly. Todeterminewhichcombinationyields
7

On-PolicySelf-DistillationforLargeLanguageModels
themosteffectivelearningsignal,weanalyzetheforward AIME25 (Qwen3-1.7B) AIME24 (Qwen3-1.7B)
| KLdivergenceKL(p |     | ∥p  | )acrossallfourstudent/teacher |     |     |                     | 46  |     |     |                        |     |
| ---------------- | --- | --- | ----------------------------- | --- | --- | ------------------- | --- | --- | --- | ---------------------- | --- |
|                  |     | T S |                               |     |     | )%( ycaruccA 21@gvA |     |     |     | )%( ycaruccA 21@gvA 60 |     |
44
| modepairings,categorizingtokensintothreegroups: |            |     |              |            | math  |     |     |     |     | 58  |     |
| ----------------------------------------------- | ---------- | --- | ------------ | ---------- | ----- | --- | --- | --- | --- | --- | --- |
| (numerals,                                      | operators, | and | mathematical | keywords), | style |     | 42  |     |     |     |     |
56
| (reasoning | connectives), |     | and other. | Table | 5 reports the |     | 40  |     |     |     |     |
| ---------- | ------------- | --- | ---------- | ----- | ------------- | --- | --- | --- | --- | --- | --- |
54
38
meanper-tokenKLwithineachcategory.
|     |     |     |     |     |     |     |     |     | Gen Length 4096 | 52  | Gen Length 4096 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --------------- | --- | --------------- |
36
Across all model sizes, the TM-off student paired with a Gen Length 1024 50 Gen Length 1024
34
|               |          |             |                   |         |             |     | 0   | 25                    | 50 75 | 100 0 25              | 50 75 100 |
| ------------- | -------- | ----------- | ----------------- | ------- | ----------- | --- | --- | --------------------- | ----- | --------------------- | --------- |
| TM-on teacher |          | yields the  | largest KL        | on math | tokens, in- |     |     |                       |       |                       |           |
|               |          |             |                   |         |             |     |     | Gradient Update Steps |       | Gradient Update Steps |           |
| dicating      | stronger | supervision | on mathematically |         | relevant    |     |     |                       |       |                       |           |
Figure5.EffectofGenerationLengthonQwen3-1.7B.Wecom-
tokens. ThereportedKLvaluescorrespondtotheexpected
parestudentgenerationlengthof1024vs4096onAIME25and
divergenceoverthevocabularyateachposition;asshown
AIME24.
inTable5,thisexpectationishighlyskewed,withstylistic
| tokens contributing                   |         | disproportionately |                               | large | values. This |            |     |       |               |          |                  |
| ------------------------------------- | ------- | ------------------ | ----------------------------- | ----- | ------------ | ---------- | --- | ----- | ------------- | -------- | ---------------- |
| motivates                             | our use | of pointwise       | clipping                      | to    | control such |            |     |       |               |          |                  |
| heavy-tailedcontributions.            |         |                    | Empirically,thisconfiguration |       |              |            |     |       |               |          |                  |
| achievesthebestdownstreamperformance. |         |                    |                               |       | Wetherefore  |            |     |       |               |          |                  |
|                                       |         |                    |                               |       |              | vocabulary |     | logit | distillation. | As shown | in Figure 5, in- |
adopttheTM-offstudent/TM-onteacherconfiguration.
creasingthegenerationlengthdoesnotleadtoconsistent
|     |     |     |     |     |     | improvementsacrosseithertask. |     |     |     | Weattributethistoearly |     |
| --- | --- | --- | --- | --- | --- | ----------------------------- | --- | --- | --- | ---------------------- | --- |
)%( ycaruccA 42EMIA 21@gvA tokensbeingmorecriticalforlearning: asthestudentgen-
58 w/o per-token KL Clipping
erationgrowslonger,latertokensbecomeincreasinglypre-
w/ per-token KL Clipping
|     | 56  |     |     |     |     | dictabletotheteacherwhenconditionedonasufficiently |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | -------------------------------------------------- | --- | --- | --- | --- | --- |
longstudentprefixsolesspenaltiesareappliedtolaterto-
54
kens. Thisphenomenonisalsonotedin(Lu&Lab,2025).
52
|     | 50  |     |     |     |     | 4.3.5.LEARNINGOBJECTIVECOMPARISON: |     |     |     |     | FULL |
| --- | --- | --- | --- | --- | --- | ---------------------------------- | --- | --- | --- | --- | ---- |
VOCABULARYLOGITSDISTILLATIONVS.
48
|     |     | 0 25 | 50  | 75  | 100 |     | SAMPLED-TOKENDISTILLATION |     |     |     |     |
| --- | --- | ---- | --- | --- | --- | --- | ------------------------- | --- | --- | --- | --- |
Gradient Update Steps
OurobjectiveinEq.6isdefinedasaper-tokendiscrepancy
Figure4.EffectofPer-TokenpointwiseKLClippingonQwen3-
|                        |     |     |                                 |     |     | betweentheteacherandstudentdistributions. |     |     |     |     | Inpractice, |
| ---------------------- | --- | --- | ------------------------------- | --- | --- | ----------------------------------------- | --- | --- | --- | --- | ----------- |
| 1.7BevaluatedonAIME24. |     |     | Clippingpreventsperformancecol- |     |     |                                           |     |     |     |     |             |
|                        |     |     |                                 |     |     | OPSDcaninstantiatethisobjectiveintwoways. |     |     |     |     | (1)Full-    |
lapse.
vocabularylogitdistillation(asinGKD(Agarwaletal.,
|     |     |     |     |     |     | 2024)): |     | for each | token position, | we compute | D(p ∥p ) |
| --- | --- | --- | --- | --- | --- | ------- | --- | -------- | --------------- | ---------- | -------- |
T S
4.3.3.EFFECTOFPER-TOKENPOINTWISECLIPPING
|     |     |     |     |     |     | over | the | entire | vocabulary | via a full softmax, | yielding a |
| --- | --- | --- | --- | --- | --- | ---- | --- | ------ | ---------- | ------------------- | ---------- |
propertoken-levelf-divergencebetweenthetwopolicies.
| As shown | in Table | 5, stylistic | tokens | can | exhibit higher |     |     |     |     |     |     |
| -------- | -------- | ------------ | ------ | --- | -------------- | --- | --- | --- | --- | --- | --- |
(2)Sampled-tokenadvantagepolicy-gradientobjective
KLdivergencethanmath-relatedtokens,causingthemto
(asintheon-policydistillationmethodofLu&Lab(2025)):
| dominate      | the training | signal.   | We  | mitigate | this issue us- |     |          |          |         |                           |                |
| ------------- | ------------ | --------- | --- | -------- | -------------- | --- | -------- | -------- | ------- | ------------------------- | -------------- |
|               |              |           |     |          |                | we  | evaluate | teacher  | and     | student log-probabilities | only at        |
| ing per-token | pointwise    | clipping. |     | As shown | in Figure 4    |     |          |          |         |                           |                |
|               |              |           |     |          |                | the | token    | actually | sampled | by the student,           | yˆ n , and use |
forQwen3-1.7B,clippingstabilizestrainingandprevents
thereverse-KLtermasascalaradvantageinsideapolicy-
performancedegradation,whichisparticularlyimportant
|     |     |     |     |     |     | gradient-styleloss. |     |     | Thus,thefirstvariantdirectlymatches |     |     |
| --- | --- | --- | --- | --- | --- | ------------------- | --- | --- | ----------------------------------- | --- | --- |
giventhatOPSDconvergesrapidlywithinahundredsteps
fulltokendistributions,whereasthesecondoptimizesanon-
oftraining.
policyRLobjectiveshapedbytheteacher’slog-probabilities
|     |     |     |     |     |     | ratherthanafull-distributiondivergence. |     |     |     |     | Wecomparethese |
| --- | --- | --- | --- | --- | --- | --------------------------------------- | --- | --- | --- | --- | -------------- |
4.3.4.EFFECTOFGENERATIONLENGTH
variantsonQwen3-4Businga2048-tokengenerationbud-
Sinceourobjectiveoperatesatthetokenlevel(Eq.6),the getduringdistillation. Table4summarizestheresults. The
numberofgeneratedtokenspersampledirectlydetermines full-vocabularydivergenceobjectiveprovidesaconsistent
the amount of supervision signal available to the student. gainoverthesampled-tokenobjective. Thissuggeststhat
Longersequencesexposethestudenttomoreteacherfeed- exposingthestudenttothefullteacherdistributionoffers
back, but they also increase computational cost and may richersupervisionthanrelyingsolelyonper-tokenon-policy
introducenoisyoruninformativecontinuations. Tostudy shaping. However,thefull-vocabularycomputationincurs
thistrade-off,weconductanablationonQwen3-1.7Bby higherpeakmemoryusageduetostoringvocabulary-sized
varying the generation length of on-policy sampled stu- logitsateveryposition,indicatingatrade-offbetweenper-
dentresponsesamong1024and4096tokensandusefull- formanceandefficiency.
8

On-PolicySelf-DistillationforLargeLanguageModels
Table4.AblationondivergencecomputationstrategiesforOPSDonQwen3-4Bwith2048generationlengthfordistillation.Wereport
pass@8accuracyonAIME25andHMMT25.Full-distributionobjectives(logitdistillation)outperformsampled-tokenobjectives.
|     | MethodVariant |     |     |     |     |     |     | AIME25 |     | HMMT25 |     |
| --- | ------------- | --- | --- | --- | --- | --- | --- | ------ | --- | ------ | --- |
OPSDw/Full-vocabularylogitdistillation(Agarwaletal.,2024) 84.1 60.0
|     | OPSDw/Sampled-tokendistillation(Lu&Lab,2025) |     |     |     |     |     |     | 82.1 |     | 57.3 |     |
| --- | -------------------------------------------- | --- | --- | --- | --- | --- | --- | ---- | --- | ---- | --- |
5.RelatedWork improvedreasoning. On-policytrainingparadigmsarealso
|                   |     |                             |     |     |     | widely used | in robotics |     | and deep | reinforcement | learning, |
| ----------------- | --- | --------------------------- | --- | --- | --- | ----------- | ----------- | --- | -------- | ------------- | --------- |
| LLMSelf-Training. |     | Ourworkconnectstoalineofre- |     |     |     |             |             |     |          |               |           |
suchasDAgger(Rossetal.,2011),whereahumanteacher
searchshowingthatLLMscanimprovebygeneratingand
providescorrectivesupervisiononthestatesvisitedbythe
exploitingtheirownsupervisionsignals(Allen-Zhu&Li,
studentpolicy.
2020;Xuetal.,2024b;Chenetal.,2024;Wangetal.,2023;
ImprovingLLMReasoningthroughSFTandRL.SFT
| Sunetal.,2023;Yuanetal.,2024;Yangetal.,2024). |     |     |     |     | Clos- |     |     |     |     |     |     |
| --------------------------------------------- | --- | --- | --- | --- | ----- | --- | --- | --- | --- | --- | --- |
estinspiritiscontextdistillation(Snelletal.,2022),which andRLaretwoprimarymethodsforimprovingLLMrea-
usesthesameunderlyingmodelasbothteacherandstudent soning ability. SFT on high-quality reasoning traces has
byprovidingtheteacherwithprivilegedcontextandthen demonstratedstrongperformance(Yuetal.,2023;LIetal.,
SFTthestudentontheteacher’sgeneratedoutputswithout 2024; Paster et al., 2023; Team, 2025a; Ye et al., 2025;
|     |     |     |     |     |     | Muennighoff | et al., | 2025; | Zhou | et al., 2023). | However, |
| --- | --- | --- | --- | --- | --- | ----------- | ------- | ----- | ---- | -------------- | -------- |
context. Thiscanbeviewedasoff-policy,wherethelearn-
ing signal is a discrete token sequence. In the reasoning priorworkshowsthatSFTcanrelyonmemorizationrather
domain, ReST (Gulcehre et al., 2023) and STaR (Zelik- thanrobustgeneralization(Chuetal.,2025). Incontrast,
RLoptimizesdirectlyforoutcome-basedobjectivescanex-
| man et al., | 2022) | similarly | rely on iterative | self-training |     |     |     |     |     |     |     |
| ----------- | ----- | --------- | ----------------- | ------------- | --- | --- | --- | --- | --- | --- | --- |
loops—generaterationalesconditionedonhintsoranswers, hibitbettergeneralization(Huanetal.,2025). Morerecent
|                   |     |                 |          |     |           | algorithms | such as | GRPO | (Guo | et al., 2025; | Shao et al., |
| ----------------- | --- | --------------- | -------- | --- | --------- | ---------- | ------- | ---- | ---- | ------------- | ------------ |
| filter by rewards |     | or ground-truth | answers, | and | fine-tune |            |         |      |      |               |              |
onsuccessfultrajectories—againyieldingharddistillation; 2024)enablescalableRLbyestimatingadvantagesfrom
Mitra&Ulukus(2025)extendsthistosoftdistillation. In- group-levelrewardswithoutrequiringanexplicitcriticasin
|     |     |     |     |     |     | PPO(Schulmanetal.,2017). |     |     | Buildingonthislineofwork, |     |     |
| --- | --- | --- | --- | --- | --- | ------------------------ | --- | --- | ------------------------- | --- | --- |
contextediting(Qietal.,2025)doeson-policysamplefrom
studentandshowsthatcontext-inducedknowledgecanbe agrowingbodyofresearchhighlightstheeffectivenessof
RLVRforreasoningtasks(Yuetal.,2025;Liuetal.,2025;
internalizedviasoftdistillationbyminimizingdivergences
anddemonstratesthisinknowledgeeditingsettings. OPSD Yueetal.,2025;Anetal.,2025;Zhengetal.,2025).
differsfromtheseapproachesinthatweperformon-policy,
| softdistillationonthestudent’sownrolloutsforreasoning |           |             |              |              |     | 6.Conclusion |     |     |     |     |     |
| ----------------------------------------------------- | --------- | ----------- | ------------ | ------------ | --- | ------------ | --- | --- | --- | --- | --- |
| tasks: the                                            | teacher’s | supervision | is per-token | distribution |     |              |     |     |     |     |     |
matchingratherthangeneratingarationaleforSFT.OPSD WeintroducedOn-PolicySelf-Distillation(OPSD),asim-
frames reasoning improvement as learning a conditional pleyeteffectiveframeworkforpost-traininglargelanguage
distributioninducedjointlybythedataset’sground-truthso- modelsonreasoningtasks. TheintuitionbehindOPSDis
thatasufficientlycapablereasoningLLMcanteachitself
| lutionsandthemodel’sownreasoningability. |     |     |     | Concurrently, |     |     |     |     |     |     |     |
| ---------------------------------------- | --- | --- | --- | ------------- | --- | --- | --- | --- | --- | --- | --- |
SDPO (Hu¨botter et al., 2026) explored similar algorithm whenithasaccesstoprivilegedinformationaboutthean-
with environment feedbacks as privilledged information swertoareasoningproblem,utilizingitsownrationalization
andSDFT(Shenfeldetal.,2026)exploredon-policyself- abilitytogradeitsweakerselfwithoutaccesstotheground
distillationoncontinuallearningtasks. truth. WeexperimentallydemonstratedthatOPSDachieves
betterperformancethanoff-policydistillation/SFT,andper-
On-PolicyDistillationmethodstrainastudentmodeldi-
formsonparwithorbetterthanGRPO,whileexhibiting
| rectly on trajectories |     | sampled | from its own | policy, | while |     |     |     |     |     |     |
| ---------------------- | --- | ------- | ------------ | ------- | ----- | --- | --- | --- | --- | --- | --- |
significantlybettersampleefficiencythanGRPO.
ateachermodelprovidesper-tokenguidancethroughKL-
basedregularizationorrelatedobjectives(Agarwaletal.,
2024;Xuetal.,2024a;Guetal.,2024;Lu&Lab,2025; 7.ImpactStatement
| Xiaomi,2026;Yangetal.,2025). |     |     | Theseapproachesmiti- |     |     |     |     |     |     |     |     |
| ---------------------------- | --- | --- | -------------------- | --- | --- | --- | --- | --- | --- | --- | --- |
Thispaperpresentsworkwhosegoalistoadvancethefield
gatedistributionshiftbyoptimizingdirectlyonthestudent’s
|     |     |     |     |     |     | ofmachinelearning. |     | Ourmethodimprovestheefficiency |     |     |     |
| --- | --- | --- | --- | --- | --- | ------------------ | --- | ------------------------------ | --- | --- | --- |
visitationdistribution,buttheytypicallyrelyonadistinct oftraininglanguagemodelsforreasoningtasks,reducing
| and often | larger | teacher model. | In this work, | we  | explore |               |       |          |     |             |               |
| --------- | ------ | -------------- | ------------- | --- | ------- | ------------- | ----- | -------- | --- | ----------- | ------------- |
|           |        |                |               |     |         | computational | costs | compared |     | to existing | reinforcement |
whetheranLLMcanteachitselfbyconditioningonmore
|     |     |     |     |     |     | learningapproaches. |     | Wedonotforeseespecificnegative |     |     |     |
| --- | --- | --- | --- | --- | --- | ------------------- | --- | ------------------------------ | --- | --- | --- |
privilegedanswerinformationandleveragingitsownrea-
societalconsequences.
soningcapabilitytoguideaweakerversionofitselftoward
9

On-PolicySelf-DistillationforLargeLanguageModels
References Hinton, G., Vinyals, O., and Dean, J. Distilling the
|          |                |     |           |               |     |     |        | knowledge | in  | a neural | network, | 2015. | URL | https: |
| -------- | -------------- | --- | --------- | ------------- | --- | --- | ------ | --------- | --- | -------- | -------- | ----- | --- | ------ |
| Agarwal, | R., Vieillard, |     | N., Zhou, | Y., Stanczyk, |     | P., | Garea, |           |     |          |          |       |     |        |
//arxiv.org/abs/1503.02531.
| S.R.,Geist,M.,andBachem,O. |     |     |                                | On-policydistillation |     |     |     |     |     |     |     |     |     |     |
| -------------------------- | --- | --- | ------------------------------ | --------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| oflanguagemodels:          |     |     | Learningfromself-generatedmis- |                       |     |     |     |     |     |     |     |     |     |     |
Hu,E.J.,Shen,Y.,Wallis,P.,Allen-Zhu,Z.,Li,Y.,Wang,
takes.InThetwelfthinternationalconferenceonlearning S.,Wang,L.,andChen,W. LoRA:Low-rankadaptation
representations,2024.
|            |        |        |         |               |     |        |     | oflargelanguagemodels. |                  |     | InInternationalConference |       |     |          |
| ---------- | ------ | ------ | ------- | ------------- | --- | ------ | --- | ---------------------- | ---------------- | --- | ------------------------- | ----- | --- | -------- |
|            |        |        |         |               |     |        |     | on Learning            | Representations, |     |                           | 2022. | URL | https:// |
| Allen-Zhu, | Z. and | Li, Y. | Towards | understanding |     | ensem- |     |                        |                  |     |                           |       |     |          |
openreview.net/forum?id=nZeVKeeFYf9.
ble, knowledgedistillationandself-distillationindeep
learning. InTheEleventhInternationalConferenceon Huan, M., Li, Y., Zheng, T., Xu, X., Kim, S., Du, M.,
LearningRepresentations,2020. Poovendran, R., Neubig, G., and Yue, X. Does math
|     |     |     |     |     |     |     |     | reasoningimprovegeneralllmcapabilities? |     |     |     |     |     | understand- |
| --- | --- | --- | --- | --- | --- | --- | --- | --------------------------------------- | --- | --- | --- | --- | --- | ----------- |
An,C.,Xie,Z.,Li,X.,Li,L.,Zhang,J.,Gong,S.,Zhong,
|                                    |     |     |     |     |     |          |     | ing transferability |     | of  | llm reasoning. |     | arXiv | preprint |
| ---------------------------------- | --- | --- | --- | --- | --- | -------- | --- | ------------------- | --- | --- | -------------- | --- | ----- | -------- |
| M.,Xu,J.,Qiu,X.,Wang,M.,andKong,L. |     |     |     |     |     | Polaris: | A   |                     |     |     |                |     |       |          |
arXiv:2507.00432,2025.
| post-training | recipe | for | scaling | reinforcement |     | learning |     |     |     |     |     |     |     |     |
| ------------- | ------ | --- | ------- | ------------- | --- | -------- | --- | --- | --- | --- | --- | --- | --- | --- |
on advanced reasoning models, 2025. URL https: Hu¨botter,J.,Lu¨beck,F.,Behric,L.,Baumann,A.,Bagatella,
//hkunlp.github.io/blog/2025/Polaris.
M.,Marta,D.,Hakimi,I.,Shenfeld,I.,KleineBuening,
|     |     |     |     |     |     |     |     | T.,Guestrin,C.,andKrause,A. |     |     |     | Reinforcementlearning |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --------------------------- | --- | --- | --- | --------------------- | --- | --- |
Chen,Z.,Deng,Y.,Yuan,H.,Ji,K.,andGu,Q. Self-play via self-distillation. arXiv preprint arXiv:2601.20802,
fine-tuningconvertsweaklanguagemodelstostronglan-
2026.
| guagemodels. |     | InInternationalConferenceonMachine |     |     |     |     |     |     |     |     |     |     |     |     |
| ------------ | --- | ---------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
Learning,pp.6621–6642.PMLR,2024. Kim,Y.andRush,A.M. Sequence-levelknowledgedistilla-
|     |     |     |     |     |     |     |     | tion. InProceedingsofthe2016conferenceonempirical |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ------------------------------------------------- | --- | --- | --- | --- | --- | --- |
Chu,T.,Zhai,Y.,Yang,J.,Tong,S.,Xie,S.,Schuurmans, methodsinnaturallanguageprocessing,pp.1317–1327,
| D.,Le,Q.V.,Levine,S.,andMa,Y. |                                     |     |     |     | Sftmemorizes,rl |     |     | 2016.             |     |               |     |             |     |            |
| ----------------------------- | ----------------------------------- | --- | --- | --- | --------------- | --- | --- | ----------------- | --- | ------------- | --- | ----------- | --- | ---------- |
| generalizes:                  | Acomparativestudyoffoundationmodel  |     |     |     |                 |     |     |                   |     |               |     |             |     |            |
|                               |                                     |     |     |     |                 |     |     | LI, J., Beeching, |     | E., Tunstall, |     | L., Lipkin, | B., | Soletskyi, |
| post-training.                | arXivpreprintarXiv:2501.17161,2025. |     |     |     |                 |     |     |                   |     |               |     |             |     |            |
R.,Huang,S.C.,Rasul,K.,Yu,L.,Jiang,A.,Shen,Z.,
Gu,Y.,Dong,L.,Wei,F.,andHuang,M. Minillm: Knowl- Qin, Z., Dong, B., Zhou, L., Fleureau, Y., Lample, G.,
|     |     |     |     |     |     |     |     | andPolu,S. | Numinamath. |     | https://github.com/ |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ---------- | ----------- | --- | ------------------- | --- | --- | --- |
edgedistillationoflargelanguagemodels.InICLR,2024.
project-numina/aimo-progress-prize/
Guha, E., Marten, R., Keh, S., Raoof, N., Smyrnis, G., blob/main/report/numina_dataset.pdf,
| Bansal,H.,Nezhurina,M.,Mercat,J.,Vu,T.,Sprague, |     |     |     |     |     |     |     | 2024. |     |     |     |     |     |     |
| ----------------------------------------------- | --- | --- | --- | --- | --- | --- | --- | ----- | --- | --- | --- | --- | --- | --- |
Z.,Suvarna,A.,Feuer,B.,Chen,L.,Khan,Z.,Frankel,
E.,Grover,S.,Choi,C.,Muennighoff,N.,Su,S.,Zhao, Lightman,H.,Kosaraju,V.,Burda,Y.,Edwards,H.,Baker,
|     |     |     |     |     |     |     |     | B., Lee, | T., Leike, | J., | Schulman, | J., | Sutskever, | I., and |
| --- | --- | --- | --- | --- | --- | --- | --- | -------- | ---------- | --- | --------- | --- | ---------- | ------- |
W.,Yang,J.,Pimpalgaonkar,S.,Sharma,K.,Ji,C.C.-J.,
|     |     |     |     |     |     |     |     |     |     |     |     |     |     | The Twelfth |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ----------- |
Deng,Y.,Pratt,S.,Ramanujan,V.,Saad-Falcon,J.,Li, Cobbe, K. Let’s verify step by step. In
J.,Dave,A.,Albalak,A.,Arora,K.,Wulfe,B.,Hegde, InternationalConferenceonLearningRepresentations,
2023.
C.,Durrett,G.,Oh,S.,Bansal,M.,Gabriel,S.,Grover,
| A., Chang,     | K.-W.,     | Shankar,                            |          | V., Gokaslan, |     | A., Merrill, |     |                       |     |                                    |     |           |     |              |
| -------------- | ---------- | ----------------------------------- | -------- | ------------- | --- | ------------ | --- | --------------------- | --- | ---------------------------------- | --- | --------- | --- | ------------ |
|                |            |                                     |          |               |     |              |     | Liu, Z., Chen,        | C., | Li, W.,                            | Qi, | P., Pang, | T., | Du, C., Lee, |
| M. A.,         | Hashimoto, | T.,                                 | Choi,    | Y., Jitsev,   | J., | Heckel,      | R., |                       |     |                                    |     |           |     |              |
|                |            |                                     |          |               |     |              |     | W.S.,andLin,M.        |     | Understandingr1-zero-liketraining: |     |           |     |              |
| Sathiamoorthy, |            | M.,                                 | Dimakis, | A. G.,        | and | Schmidt,     | L.  |                       |     |                                    |     |           |     |              |
|                |            |                                     |          |               |     |              |     | Acriticalperspective. |     | arXivpreprintarXiv:2503.20783,     |     |           |     |              |
| Openthoughts:  |            | Datarecipesforreasoningmodels,2025. |          |               |     |              |     |                       |     |                                    |     |           |     |              |
2025.
URLhttps://arxiv.org/abs/2506.04178.
|             |            |         |                 |           |                  |        |     | Loshchilov,I.andHutter,F. |                                     |     | Decoupledweightdecayregu- |     |     |     |
| ----------- | ---------- | ------- | --------------- | --------- | ---------------- | ------ | --- | ------------------------- | ----------------------------------- | --- | ------------------------- | --- | --- | --- |
| Gulcehre,   | C., Paine, | T.      | L., Srinivasan, |           | S., Konyushkova, |        |     |                           |                                     |     |                           |     |     |     |
|             |            |         |                 |           |                  |        |     | larization.               | arXivpreprintarXiv:1711.05101,2017. |     |                           |     |     |     |
| K., Weerts, | L.,        | Sharma, | A.,             | Siddhant, | A.,              | Ahern, | A., |                           |                                     |     |                           |     |     |     |
Wang,M.,Gu,C.,etal. Reinforcedself-training(rest) Lu, K. and Lab, T. M. On-policy distillation. Thinking
| forlanguagemodeling. |     |     | arXivpreprintarXiv:2308.08998, |     |     |     |     |              |                                             |                     |     |     |                    |     |
| -------------------- | --- | --- | ------------------------------ | --- | --- | --- | --- | ------------ | ------------------------------------------- | ------------------- | --- | --- | ------------------ | --- |
|                      |     |     |                                |     |     |     |     | MachinesLab: |                                             | Connectionism,2025. |     |     | doi: 10.64434/tml. |     |
| 2023.                |     |     |                                |     |     |     |     | 20251026.    | https://thinkingmachines.ai/blog/on-policy- |                     |     |     |                    |     |
distillation.
Guo,D.,Yang,D.,Zhang,H.,Song,J.,Zhang,R.,Xu,R.,
Zhu,Q.,Ma,S.,Wang,P.,Bi,X.,etal. Deepseek-r1: In- Mitra,P.andUlukus,S. Semanticsoftbootstrapping: Long
centivizingreasoningcapabilityinllmsviareinforcement contextreasoninginllmswithoutreinforcementlearning.
learning. arXivpreprintarXiv:2501.12948,2025. arXivpreprintarXiv:2512.05105,2025.
10

On-PolicySelf-DistillationforLargeLanguageModels
Muennighoff,N.,Yang,Z.,Shi,W.,Li,X.L.,Fei-Fei,L., Sun, Z., Yu, L., Shen, Y., Liu, W., Yang, Y., Welleck, S.,
Hajishirzi, H., Zettlemoyer, L., Liang, P., Cande`s, E., andGan,C. Easy-to-hardgeneralization: Scalablealign-
andHashimoto,T. s1: Simpletest-timescaling. arXiv ment beyond human supervision. Advances in Neural
preprintarXiv:2501.19393,2025. InformationProcessingSystems,37:51118–51168,2024.
Naor, M. Evaluation may be easier than generation. In Team, K., Bai, Y., Bao, Y., Chen, G., Chen, J., Chen,
|     |     |     |     |     | N., Chen, | R., | Chen, | Y., Chen, | Y., | Chen, | Y., et al. |
| --- | --- | --- | --- | --- | --------- | --- | ----- | --------- | --- | ----- | ---------- |
Proceedingsofthetwenty-eighthannualACMsymposium
onTheoryofcomputing,pp.74–83,1996. Kimi k2: Open agentic intelligence. arXiv preprint
arXiv:2507.20534,2025.
| Paster,K.,Santos,M.D.,Azerbayev,Z.,andBa,J. |                                         |     |     | Open- |                       |     |     |                                  |     |     |     |
| ------------------------------------------- | --------------------------------------- | --- | --- | ----- | --------------------- | --- | --- | -------------------------------- | --- | --- | --- |
|                                             |                                         |     |     |       | Team,O. OpenThoughts. |     |     | https://open-thoughts.ai,January |     |     |     |
| webmath:                                    | Anopendatasetofhigh-qualitymathematical |     |     |       |                       |     |     |                                  |     |     |     |
2025a.
webtext,2023.
|     |     |     |     |     | Team, Q. | Qwen3technicalreport, |     |     | 2025b. | URLhttps: |     |
| --- | --- | --- | --- | --- | -------- | --------------------- | --- | --- | ------ | --------- | --- |
Qi,S.,Yang,B.,Jiang,K.,Wang,X.,Li,J.,Zhong,Y.,Yang,
//arxiv.org/abs/2505.09388.
| Y., and                            | Zheng, Z. In-context |     | editing:        | Learning knowl- |           |        |             |     |      |            |        |
| ---------------------------------- | -------------------- | --- | --------------- | --------------- | --------- | ------ | ----------- | --- | ---- | ---------- | ------ |
| edgefromself-induceddistributions. |                      |     | InTheThirteenth |                 |           |        |             |     |      |            |        |
|                                    |                      |     |                 |                 | Wang, Y., | Kordi, | Y., Mishra, | S., | Liu, | A., Smith, | N. A., |
InternationalConferenceonLearningRepresentations,
|       |     |     |     |     | Khashabi,D.,andHajishirzi,H. |        |      |                | Self-instruct: |               | Aligning |
| ----- | --- | --- | --- | --- | ---------------------------- | ------ | ---- | -------------- | -------------- | ------------- | -------- |
| 2025. |     |     |     |     | language                     | models | with | self-generated |                | instructions. | In       |
Proceedingsofthe61stannualmeetingoftheassociation
| Rastogi, | A., Jiang, A. | Q., Lo, A., | Berrada, | G., Lample, |     |     |     |     |     |     |     |
| -------- | ------------- | ----------- | -------- | ----------- | --- | --- | --- | --- | --- | --- | --- |
forcomputationallinguistics(volume1:longpapers),pp.
| G., Rute, | J., Barmentlo, | J., | Yadav, | K., Khandelwal, |     |     |     |     |     |     |     |
| --------- | -------------- | --- | ------ | --------------- | --- | --- | --- | --- | --- | --- | --- |
13484–13508,2023.
| K., Chandu, | K. R., | et al. Magistral. |     | arXiv preprint |     |     |     |     |     |     |     |
| ----------- | ------ | ----------------- | --- | -------------- | --- | --- | --- | --- | --- | --- | --- |
arXiv:2506.10910,2025. Xiaomi,L.-C. Mimo-v2-flashtechnicalreport,2026. URL
https://arxiv.org/abs/2601.02780.
| Ross,S.,Gordon,G.,andBagnell,D. |     |     | Areductionofimita- |     |     |     |     |     |     |     |     |
| ------------------------------- | --- | --- | ------------------ | --- | --- | --- | --- | --- | --- | --- | --- |
tionlearningandstructuredpredictiontono-regretonline Xu,W.,Han,R.,Wang,Z.,Le,L.,Madeka,D.,Li,L.,Wang,
learning. InProceedingsofthefourteenthinternational W.Y.,Agarwal,R.,Lee,C.-Y.,andPfister,T. Speculative
| conference | on artificial | intelligence | and | statistics, |                        |     |     |                               |     |     |     |
| ---------- | ------------- | ------------ | --- | ----------- | ---------------------- | --- | --- | ----------------------------- | --- | --- | --- |
|            |               |              |     | pp.         | knowledgedistillation: |     |     | Bridgingtheteacher-studentgap |     |     |     |
627–635.JMLRWorkshopandConferenceProceedings, throughinterleavedsampling. InTheThirteenthInterna-
| 2011. |     |     |     |     | tionalConferenceonLearningRepresentations,2024a. |     |     |     |     |     |     |
| ----- | --- | --- | --- | --- | ------------------------------------------------ | --- | --- | --- | --- | --- | --- |
Xu,X.,Li,M.,Tao,C.,Shen,T.,Cheng,R.,Li,J.,Xu,C.,
| Sanh,V.,Debut,L.,Chaumond,J.,andWolf,T. |     |     |     | Distilbert, |     |     |     |     |     |     |     |
| --------------------------------------- | --- | --- | --- | ----------- | --- | --- | --- | --- | --- | --- | --- |
a distilled version of bert: smaller, faster, cheaper and Tao,D.,andZhou,T. Asurveyonknowledgedistillation
lighter. arXivpreprintarXiv:1910.01108,2019. oflargelanguagemodels. CoRR,2024b.
Yang,A.,Li,A.,Yang,B.,Zhang,B.,Hui,B.,Zheng,B.,
| Schulman, | J., Wolski, | F., Dhariwal, | P., | Radford, A., and |     |     |     |     |     |     |     |
| --------- | ----------- | ------------- | --- | ---------------- | --- | --- | --- | --- | --- | --- | --- |
Yu,B.,Gao,C.,Huang,C.,Lv,C.,Zheng,C.,Liu,D.,
| Klimov, | O. Proximal | policy | optimization | algorithms. |     |     |     |     |     |     |     |
| ------- | ----------- | ------ | ------------ | ----------- | --- | --- | --- | --- | --- | --- | --- |
arXivpreprintarXiv:1707.06347,2017. Zhou,F.,Huang,F.,Hu,F.,Ge,H.,Wei,H.,Lin,H.,Tang,
|     |     |     |     |     | J., Yang, | J., Tu, | J., Zhang, | J., | Yang, | J., Yang, | J., Zhou, |
| --- | --- | --- | --- | --- | --------- | ------- | ---------- | --- | ----- | --------- | --------- |
Shao,Z.,Wang,P.,Zhu,Q.,Xu,R.,Song,J.,Bi,X.,Zhang, J.,Zhou,J.,Lin,J.,Dang,K.,Bao,K.,Yang,K.,Yu,L.,
H.,Zhang,M.,Li,Y.,Wu,Y.,etal. Deepseekmath: Push- Deng,L.,Li,M.,Xue,M.,Li,M.,Zhang,P.,Wang,P.,
ingthelimitsofmathematicalreasoninginopenlanguage Zhu,Q.,Men,R.,Gao,R.,Liu,S.,Luo,S.,Li,T.,Tang,
models. arXivpreprintarXiv:2402.03300,2024. T.,Yin,W.,Ren,X.,Wang,X.,Zhang,X.,Ren,X.,Fan,
Y.,Su,Y.,Zhang,Y.,Zhang,Y.,Wan,Y.,Liu,Y.,Wang,
Shenfeld, I., Damani, M., Hu¨botter, J., and Agrawal, P. Z., Cui, Z., Zhang, Z., Zhou, Z., and Qiu, Z. Qwen3
| Self-distillationenablescontinuallearning,2026. |     |     |     | URL |                  |     |                                     |     |     |     |     |
| ----------------------------------------------- | --- | --- | --- | --- | ---------------- | --- | ----------------------------------- | --- | --- | --- | --- |
|                                                 |     |     |     |     | technicalreport. |     | arXivpreprintarXiv:2505.09388,2025. |     |     |     |     |
https://arxiv.org/abs/2601.19897.
Yang,Z.,Pang,T.,Feng,H.,Wang,H.,Chen,W.,Zhu,M.,
Snell,C.,Klein,D.,andZhong,R. Learningbydistilling andLiu,Q. Self-distillationbridgesdistributiongapin
| context. | arXivpreprintarXiv:2209.15189,2022. |     |     |     |                           |         |        |                        |     |               |     |
| -------- | ----------------------------------- | --- | --- | --- | ------------------------- | ------- | ------ | ---------------------- | --- | ------------- | --- |
|          |                                     |     |     |     | languagemodelfine-tuning. |         |        | InProceedingsofthe62nd |     |               |     |
|          |                                     |     |     |     | Annual                    | Meeting | of the | Association            | for | Computational |     |
Sun,Z.,Shen,Y.,Zhou,Q.,Zhang,H.,Chen,Z.,Cox,D.,
|       |                 |                  |     |                | Linguistics | (Volume | 1:  | Long | Papers), | pp. 1028–1043, |     |
| ----- | --------------- | ---------------- | --- | -------------- | ----------- | ------- | --- | ---- | -------- | -------------- | --- |
| Yang, | Y., and Gan, C. | Principle-driven |     | self-alignment |             |         |     |      |          |                |     |
2024.
| of language | models | from scratch | with | minimal human |     |     |     |     |     |     |     |
| ----------- | ------ | ------------ | ---- | ------------- | --- | --- | --- | --- | --- | --- | --- |
supervision. In Thirty-seventh Conference on Neural Ye,Y.,Huang,Z.,Xiao,Y.,Chern,E.,Xia,S.,andLiu,P.
Information Processing Systems, 2023. URL https: Limo: Lessismoreforreasoning,2025. URLhttps:
//openreview.net/forum?id=p40XRfBX96. //arxiv.org/abs/2502.03387.
11

On-PolicySelf-DistillationforLargeLanguageModels
Yu,L.,Jiang,W.,Shi,H.,Yu,J.,Liu,Z.,Zhang,Y.,Kwok,
| J. T., Li, | Z., Weller, | A., and | Liu, W. | Metamath: | Boot- |
| ---------- | ----------- | ------- | ------- | --------- | ----- |
strapyourownmathematicalquestionsforlargelanguage
| models. | arXivpreprintarXiv:2309.12284,2023. |     |     |     |     |
| ------- | ----------------------------------- | --- | --- | --- | --- |
Yu,Q.,Zhang,Z.,Zhu,R.,Yuan,Y.,Zuo,X.,Yue,Y.,Fan,
| T.,Liu,G.,Liu,L.,Liu,X.,etal.               |     |     | Dapo: | Anopen-source |     |
| ------------------------------------------- | --- | --- | ----- | ------------- | --- |
| llmreinforcementlearningsystematscale,2025. |     |     |       |               | URL |
https://arxiv.org/abs/2503.14476,2025.
Yuan,W.,Pang,R.Y.,Cho,K.,Li,X.,Sukhbaatar,S.,Xu,
| J., and Weston,  | J. E.      | Self-rewarding |            | language  | models. |
| ---------------- | ---------- | -------------- | ---------- | --------- | ------- |
| In International | Conference |                | on Machine | Learning, | pp.     |
57905–57923.PMLR,2024.
Yue,Y.,Yuan,Y.,Yu,Q.,Zuo,X.,Zhu,R.,Xu,W.,Chen,
| J.,Wang,C.,Fan,T.,Du,Z.,etal. |     |     |     | Vapo: Efficientand |     |
| ----------------------------- | --- | --- | --- | ------------------ | --- |
reliablereinforcementlearningforadvancedreasoning
tasks. arXivpreprintarXiv:2504.05118,2025.
| Zelikman,E.,Wu,Y.,Mu,J.,andGoodman,N. |     |     |     |                  | Star: Boot- |
| ------------------------------------- | --- | --- | --- | ---------------- | ----------- |
| strappingreasoningwithreasoning.      |     |     |     | AdvancesinNeural |             |
InformationProcessingSystems,35:15476–15488,2022.
Zhang,Z.,Zheng,C.,Wu,Y.,Zhang,B.,Lin,R.,Yu,B.,
| Liu,D.,Zhou,J.,andLin,J.                    |     |     | Thelessonsofdeveloping |     |       |
| ------------------------------------------- | --- | --- | ---------------------- | --- | ----- |
| processrewardmodelsinmathematicalreasoning. |     |     |                        |     | arXiv |
preprintarXiv:2501.07301,2025.
| Zhao, S.,                               | Liu, M., Huang, | J., | Liu, M., | Wang, | C., Liu, B., |
| --------------------------------------- | --------------- | --- | -------- | ----- | ------------ |
| Tian,Y.,Pang,G.,Bell,S.,Grover,A.,etal. |                 |     |          |       | Inpainting-  |
guidedpolicyoptimizationfordiffusionlargelanguage
| models.        | arXivpreprintarXiv:2509.10396,2025. |               |              |       |            |
| -------------- | ----------------------------------- | ------------- | ------------ | ----- | ---------- |
| Zheng, C.,     | Liu, S., Li,                        | M.,           | Chen, X.-H., | Yu,   | B., Gao,   |
| C., Dang,      | K., Liu,                            | Y., Men,      | R.,          | Yang, | A., et al. |
| Group sequence | policy                              | optimization. |              | arXiv | preprint   |
arXiv:2507.18071,2025.
| Zhou, C.,     | Liu, P., Xu,                        | P., Iyer, | S., Sun, | J., Mao, | Y., Ma,      |
| ------------- | ----------------------------------- | --------- | -------- | -------- | ------------ |
| X., Efrat,    | A., Yu, P.,                         | Yu, L.,   | et al.   | Lima:    | less is more |
| foralignment. | InProceedingsofthe37thInternational |           |          |          |              |
ConferenceonNeuralInformationProcessingSystems,
pp.55006–55021,2023.
12

On-PolicySelf-DistillationforLargeLanguageModels
A.LimitationsandFutureDirections
Duetocomputationalconstraints,ourexperimentsarelimitedtomodelsupto8Bparameters. Itremainsanopenquestion
whetherthistrendcontinuesatscalesbeyond8Bparameters. Severalpromisingdirectionswarrantfurtherinvestigation.
First,ourcurrentframeworkdoesnotexplicitlyleveragecorrectnessverificationofgeneratedanswers;incorporatingsuch
signals could provide additional learning objectives beyond distribution matching. Finally, problem difficulty plays a
crucial role in self-distillation: if reasoning problems exceed the model’s comprehension threshold, the teacher policy
cannotprovidemeaningfulsupervisionevenwithaccesstoground-truthsolutions. Thissuggeststhatcurriculumlearning
strategies—graduallyincreasingproblemdifficultyasthemodelimproves—couldenhancetrainingeffectiveness. Exploring
adaptivecurriculathatmaintainproblemsatthefrontierofmodelcapabilitiesrepresentsanimportantdirectionforscaling
OPSDtomorechallengingreasoningtasks.
B.ExperimentalDetails
Table5.Per-tokenKLdivergencebytokencategoryacrossgenerationstyles.Meanper-tokenKLdivergencebrokendownbytoken
category(seeAppendixCfordetaileddefinitions),averagedover10problems.ThinkingModeOFF/ONindicateswhetherthestudentor
teacherLLM’spromptformatenablesthinkingmode.Wefindwhenstudent’sgeneration’sthinkingmodeisoffandwhentheteacher’s
thinkingmodeison,theKLsignalonmathrelatedtokensarethehighest.Andwechoosethissetupforourexperiments.
|                 | Qwen3-1.7B |       | Qwen3-4B         | Qwen3-8B   |       |
| --------------- | ---------- | ----- | ---------------- | ---------- | ----- |
| Student Teacher | Style Math | Other | Style Math Other | Style Math | Other |
| TM-off TM-off   | 0.68 0.12  | 0.11  | 0.61 0.06 0.10   | 0.56 0.05  | 0.11  |
| TM-on TM-off    | 0.51 0.10  | 0.17  | 0.41 0.05 0.18   | 0.33 0.05  | 0.15  |
| TM-on TM-on     | 0.51 0.09  | 0.08  | 0.50 0.04 0.09   | 0.42 0.04  | 0.08  |
| TM-off TM-on    | 0.85 0.14  | 0.25  | 0.92 0.10 0.29   | 0.79 0.06  | 0.25  |
WeprovidethetrainingandevaluationconfigurationsforourSFT,GRPOandOPSDexperimentsinTables7,6and8.
NotethatweadopttheThinking-Mode-offstudent/Thinking-Mode-onteacherconfigurationformainOPSDexperiments.
Formoreexperimentdetails,pleaserefertoourreleasedtrainingcodeinhttps://github.com/siyan-zhao/OPSD.Wedidn’t
conducttuningfortheclippingparameterτ,optimizingthishyperparametermayyieldfurtherperformancegainswithinthe
same100-stepbudgetforlargermodels.
Table6.TrainingConfigurationforGRPOandOPSD
| Parameter |     |     | GRPO   | OPSD   |     |
| --------- | --- | --- | ------ | ------ | --- |
|           |     |     | 5×10−6 | 5×10−6 |     |
LearningRate
| EffectiveBatchSize           |     |     | 32              | 32        |       |
| ---------------------------- | --- | --- | --------------- | --------- | ----- |
| LoRARank(r)                  |     |     | 64              | 64        |       |
| LoRAAlpha(α)                 |     |     | 128             | 128       |       |
| LoRATargetModules            |     |     | q proj,k proj,v | proj,o    | proj, |
|                              |     |     | gate proj,up    | proj,down | proj  |
| MaxCompletionLength          |     |     | 16,000          | 1024      |       |
| NumberofGenerationsperPrompt |     |     | 8               | 1         |       |
| SamplingTemperature          |     |     | 1.2             | 1.1       |       |
| KLCoefficient(β)             |     |     | 0.0             | –         |       |
| TrainingSteps                |     |     | 500             | 100       |       |
Allexperimentswereconductedusing8A100orH100GPUswithgradientcheckpointingandFlashAttention2formemory
efficiency. WeusetheAdamW(Loshchilov&Hutter,2017)optimizerandbfloat16precisionforalltrainingruns. For
OPSD,unlessotherwisestated,weusedfull-vocabularylogitdistillation.
13

On-PolicySelf-DistillationforLargeLanguageModels
Table7.TrainingConfigurationforSFT.
| Parameter            |                 | SFT            |
| -------------------- | --------------- | -------------- |
| LearningRate         | 5×10−6          |                |
| EffectiveBatchSize   |                 | 32             |
| LoRARank(r)          |                 | 64             |
| LoRAAlpha(α)         |                 | 128            |
| LoRATargetModules    | q proj,k proj,v | proj,o proj,   |
|                      | gate proj,up    | proj,down proj |
| MaxSequenceLength    | 16000           |                |
| NumberofTrainingStep |                 | 100            |
Table8.EvaluationParameters.
Parameter Value
MaxNewTokens 38912
ThinkingMode Enabled
Top-p 0.95
Top-k -1
Min-p 0.0
PresencePenalty 0.0
SamplesperPrompt 12
Temperature 1.0
C.TokenCategoryDefinitions
Wecategorizetokensintostyleandmathgroupsusingpredefinedkeywordlists. Thesekeywordsetsareusedtoanalyzethe
per-tokenKLdivergencestylistictokensandmathematicalknowledgetokensasinSection4.3.1.
Style Tokens. maybe, perhaps, probably, possibly, let, okay, ok, alright, hmm, wait, because, since, so, thus, hence,
therefore,but,however,although,though,yet,or,alternatively,instead,otherwise,actually,really,just,simply,basically,
very, quite, pretty, rather, fairly, now, then, next, first, second, finally, try, see, check, note, recall, think, idea, strategy,
approach,method,way,would,could,should,might,can,huge,large,big,small,tiny,interesting,tricky,complex,simple.
MathTokens. exponential,exponent,power,powers,base,logarithm,logarithms,log,ln,compare,comparing,compari-
son,less,equal,larger,smaller,greater,factor,factors,prime,divisible,equation,expression,formula,inequality,rational,
irrational,real,integer,coefficient,variable,constant,sum,product,difference,quotient,fraction,denominator,numerator,
root,square,cube,nth,maximum,minimum,optimize,bound.
D.Policy-GradientInterpretationofOPSDandComparisontoSTaR
OurOPSDobjectiveinEquation(9)canbeinterpretedasapolicy-gradientupdatewithadense,token-levelrewardsignal
derivedfromprivilegedinformation. Inthissection,weshow: (1)OPSDcanbeseenasadense-rewardpolicygradient,and
(2)wecontrastOPSDwithSTaR,demonstratingthatSTaR’slearningsignalissequence-levelwhileOPSDistoken-level.
D.1.STaRasSequence-LevelPolicy-Gradient
STaR(Zelikmanetal.,2022)canbeviewedasanapproximationtoanRL-stylepolicygradientobjective. Thelanguage
modelp inducesajointdistributionoverrationalerandanswery:
θ
| p (r,y |x)=p | (r |x)p (y |x,r), |     |
| ------------ | ----------------- | --- |
θ θ θ
14

On-PolicySelf-DistillationforLargeLanguageModels
wherethemodelfirstsamplesalatentrationalerbeforepredictingthefinalanswery. GivenanindicatorrewardR(y)=
| 1(y =y⋆),theexpectedreturnacrossthedatasetS |     |     |     |     | ={(x | ,y⋆)}N | is  |     |     |     |     |     |
| ------------------------------------------- | --- | --- | --- | --- | ---- | ------ | --- | --- | --- | --- | --- | --- |
|                                             |     |     |     |     |      | i i    | i=1 |     |     |     |     |     |
(cid:88) N
|     |     |     |     |             | E   |                | (cid:2) | =y⋆) | (cid:3) |     |     |      |
| --- | --- | --- | --- | ----------- | --- | -------------- | ------- | ---- | ------- | --- | --- | ---- |
|     |     |     |     | J STaR (θ)= |     | (r,y)∼pθ(·|xi) | 1(y     |      | .       |     |     | (10) |
i
i=1
Applyingthelog-derivativetrickyieldsapolicygradient:
N
|     |     |     |        | (cid:88) |                | (cid:104) |       |      |      | (cid:105) |     |      |
| --- | --- | --- | ------ | -------- | -------------- | --------- | ----- | ---- | ---- | --------- | --- | ---- |
|     |     | ∇   | J (θ)= | E        |                | 1(y       | =y⋆)∇ | logp | (r,y | |x )      | .   | (11) |
|     |     | θ   | STaR   |          | (r,y)∼pθ(·|xi) |           | i     | θ    | θ    | i         |     |      |
i=1
Notethattheindicatorfunctiondiscardsthegradientforallsampledrationalesthatdonotleadtothecorrectanswery⋆: this
i
correspondstothefilteringstepinSTaR.
OnelimitationisthatSTaR’srewardissequence-level: thebinaryindicator1(y =y⋆)providesthesamesignaltoalltokens
inatrajectory,offeringnointermediatecreditassignment. Whenallsampledtrajectoriesareallincorrect,thelearningsignal
vanishes.
D.2.OPSDasDense-RewardPolicyGradient
Thesampled-tokenobjectiveinEquation(9)canalsobeviewedasapolicy-gradientmethod,butwithatoken-levelreward.
Fixatrainingpair(x,y⋆)andletthestudentgenerateatrajectoryyˆ∼p (·|x). Ateachpositionn,definetheper-token
S
reward:
|     |     |     | (x,yˆ)≜logp |     | |x,y⋆,yˆ |     |        |       |            |     |     |     |
| --- | --- | --- | ----------- | --- | -------- | --- | ------ | ----- | ---------- | --- | --- | --- |
|     |     |     | r n         |     | T (yˆ n  | <n  | )−logp | S (yˆ | n |x,yˆ <n | ).  |     |     |
Thisrewardmeasureshowmuchtheprivilegedteacherprefersthesampledtokenyˆ relativetothestudent. Asstatedinthe
n
maintext,wetreatr n (equivalently,theadvantageA n )asaconstantwithrespecttoθwhencomputinggradients—thatis,
westopgradientsthroughbothp andp intherewardcomputation. Underthistreatment,thegradientofEquation(9)
|     |     |     | T   | S   |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
takesthestandardpolicy-gradientform:
|     |     |     |     |    |     |    |     |     |     |     |   |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|yˆ|
1 (cid:88)
|     | ∇   | L(θ)=−E |           | E   |             |     | r (x,yˆ)∇ |     | logp (yˆ | |x,yˆ | ), |     |
| --- | --- | ------- | --------- | --- | ----------- | --- | --------- | --- | -------- | ----- | ---- | --- |
|     | θ   |         | (x,y⋆)∼S |     | yˆ∼pS(·|x) |     | n         | θ   | S        | n     | <n   |     |
|yˆ|
n=1
whichcorrespondstomaximizingtheexpectedper-tokenrewardalongon-policystudentrollouts:
|     |     |     |     |     |    |     |    |     |     |   |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|yˆ|
1 (cid:88)
|     |     |     |        | (θ)=E |           | E           |     |     |               |     |     |     |
| --- | --- | --- | ------ | ----- | --------- | ----------- | --- | --- | ------------- | --- | --- | --- |
|     |     |     | J OPSD |       | (x,y⋆)∼S | yˆ∼pS(·|x) |     |     | r n (x,yˆ). |     |     |     |
|yˆ|
n=1
Thisrewardisdense: itprovidesalearningsignalateverytokenposition,regardlessofwhetherthefinalansweriscorrect.
Comparison. BothSTaRandOPSDcanbeunderstoodaspolicy-gradientmethods,buttheirrewardstructuresdiffer
STaRusesasequence-levelindicator1(y =y⋆)thatassignsthesamesignaltoalltokens;whenallsampled
fundamentally.
trajectoriesareincorrect,thelearningsignalvanishesentirely. Incontrast,OPSDprovidesatoken-levelrewardr atevery
n
position,enablingfine-grainedcreditassignmentevenwhenthefinalansweriswrong.
15