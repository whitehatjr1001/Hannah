Learning-based Multi-agent Race Strategies in Formula 1
Giona Fieni1
, Joschua Wuthrich ¨
1
, Marc-Philippe Neumann1
, Christopher H. Onder1
Abstract— In Formula 1, race strategies are adapted according to evolving race conditions and competitors’ actions. This
paper proposes a reinforcement learning approach for multiagent race strategy optimization. Agents learn to balance energy
management, tire degradation, aerodynamic interaction, and
pit-stop decisions. Building on a pre-trained single-agent policy,
we introduce an interaction module that accounts for the behavior of competitors. The combination of the interaction module
and a self-play training scheme generates competitive policies,
and agents are ranked based on their relative performance.
Results show that the agents adapt pit timing, tire selection,
and energy allocation in response to opponents, achieving robust
and consistent race performance. Because the framework relies
only on information available during real races, it can support
race strategists’ decisions before and during races.
I. INTRODUCTION
Formula 1 (F1) is the most famous motorsport. Each year,
22 drivers from 11 teams compete in over 20 races. Teams
optimize every aspect of performance, from car development
and power unit (PU) operation to strategic planning before
and during races. Drivers, in turn, must consistently extract
maximum performance while minimizing errors. However,
race conditions often deviate from predictions, making rapid
decision-making under uncertainty essential. In such situations, the experience of race engineers is extremely important.
Since 2014, F1 has adopted hybrid-electric PUs. The
internal combustion engine (ICE) and the motor-generator
unit – kinetic (MGU-K) operate in synergy to generate
the propulsive power. In addition to fuel management, the
electrical energy stored in the battery must be carefully
deployed. As fuel mass directly affects vehicle performance
– lighter cars are faster – energy allocation becomes a critical
performance factor.
Race strategies mostly influence the race outcome. Regulations require the use of at least two different tire compounds
during a race. When tire performance deteriorates, the pit
wall calls for a pit stop. The performance gain of new tires
must compensate the time lost in the pit lane.
Despite extensive simulations, it is impossible to predict
every scenario. Teams typically compute a large number of
Monte Carlo simulations to prepare three to four baseline
strategies, which are continuously adapted according to the
evolving race context. This includes observing opponents’
performance and react to their strategic decisions.
In this paper, our goal is to support the online decisionmaking process with algorithms. In F1, decisions have to be
taken within seconds: Rather than predicting, the focus is to
1
Institute for Dynamic Systems and Control, ETH Zurich, Z ¨ urich, ¨
Switzerland, gfieni@ethz.ch
robustly react to unforeseen events while accounting for the
active response of a competitor.
A. Related work
For the research literature related to this paper, we consider
single-agent race strategies and its multi-agent perspective.
The first part can be subdivided in simulations [1], optimizations [2]–[6], and learning-based methods [7]–[10]. In
[1], neural networks (NNs) used as virtual strategy engineer
in race simulations delivered results close to reality. Optimizations deal with energy management [2], charging stops
in endurance racing [3], evolutionary algorithms [4] and codesign [6] for pit stops. Stochastic scenarios optimized via
dynamic programming (DP) are studied in [5]. Learningbased methods mainly employ reinforcement learning (RL)
to tackle the problem. In particular, [10] considers both
energy management and pit stops.
When considering competitors’ interaction, it has to be
distinguished between competitors’ awareness and multiagent methods. Building on [11], a model predictive control
(MPC) framework for competitor-aware race strategies is
developed in [12]. A two-player game is solved in [13] with
DP. While pit stops are optimized, the energy management is
neglected. Multi-agent RL is explored in [14] for Formula E
races. However, different regulations and requirements render
the problem substantially different: The racing model is
simplified, pit stops are not driven by tire degradation and
the performance of different tire compounds is neglected.
We aim to bridge the gap towards multi-agent race strategies in F1 accounting for active competitors’ response. In
particular, the framework incorporates energy management,
pit stops, and tire degradation. Given the non-smooth and
multi-agent system, classical optimization methods encounter
practical limitations, whereas RL emerges as promising tool.
Our approach builds upon [10], where the nominal policy is
benchmarked against an optimization framework.
B. Contributions
This paper provides the following contributions. First, we
develop a framework for the training and simulation of multiagent race strategies. In particular, we bridge the gap between
the single-agent RL approach of [10] and competitor-reacting
agents. By extending the policy with an interaction module
and adopting a self-play training scheme, we account for the
active response of a competitor.
Second, we show that the generated agents adapt their
race strategy according to the opponents, in terms of energy
management, pit stops and tire compound selection. Because
the agents rely only on information available during real
arXiv:2602.23056v1 [cs.AI] 26 Feb 2026
Agent 1
Environment
Race car 1 Aerodynamic
interaction Race car i Agent i
a
1
o
1
˜o
i
˜o
1
o
i
a
i
T
1
lap
t
1
gap, ∆T
1
int
T
i
lap
t
i
gap, ∆T
i
int
Fig. 1. Schematic of the agent-environment interaction. Agent 1 is the agent to be trained, while agent i is fixed and it is part of the environment. With
their actions, they directly affect the ego car. Aerodynamic interaction couples the models of the two cars. The observations are divided whether they come
from the ego car or from the competitor’s one.
races, their outputs can support race strategist to improve
their decision-making process.
C. Outline
This paper is organized as follows: Section II introduces
the RL environment, the interaction model and the agent
structure that accounts for interactions. Section III presents
the extension to the multi-agent framework, including the
training scheme and the generation of high-performing
agents. Finally, in Section IV we let the agents compete with
each other and we conclude the paper in Section V.
II. REINFORCEMENT LEARNING SETUP
In this section, we describe the setup for the RL framework. First, we present the environment and its submodules.
Then, we motivate the choice for the agent’s architecture and
formulate the Markov decision process (MDP).
A. Environment
Building upon [10], we start from the same setup and
we extend the framework to handle multi-agent interactions.
One episode corresponds to a race, which is discretized on a
lap-by-lap basis. We denote with k ∈ [0, . . . , Nlaps] the lap
number.
Figure 1 shows a schematic of the environment and the information exchange. Agent 1 interacts with the environment
using the action vector a
1
, and it receives the observations o
1
and ˜o
i
. To include the active response of agent i embedded
in the environment, actions are sampled from its policy based
on its observations.
The race strategy is the collection of pit wall’s decisions
taken for the car. These are the allocated fuel and battery
energy per lap, respectively ∆Ef,all and ∆Eb,all, as well as
the pit stop and tire compound decision, which are captured
by the decision variable PS ∈ {0, 1, 2, 3}, defined in Table I.
In our setup, the agent corresponds to the pit wall. For
this reason, the action vector is
a =

∆Ef,all ∆Eb,all PS
, (1)
which influences directly only the states of the ego car. The
dynamics of the single cars evolve accordingly to the blocks
Race car, presented in Section II-B.
PS 0 1 2 3
Action do not pit pit for soft pit for medium pit for hard
TABLE I
ACTIONS CORRESPONDING TO THE PS VARIABLE.
An agent cannot fully observe the states of the opponent.
To potentially deploy the agent in real races, only information
available to the pit wall of the team can be exploited. Hence,
we distinguish between the internal states of the ego car o
and the observable information about the competitor ˜o, both
defined in Section II-B.
Racing in the wake of another car affects the strategy.
The inclusion of an interaction model allows to investigate
alternative strategies that explicitly account for the presence
of another agent. We capture slipstream effects with the block
Aerodynamic interaction, presented in Section II-C.
B. Race car model
This model describes the evolution of the car’s internal
states according to the action given by the pit wall, as shown
in Figure 2. We adopt the same model equations of [10] and
consider the following states: the battery energy content Eb,
the available fuel energy Ef
, the car mass changing due to
fuel consumption mcar, the tire compound TC, the tire wear
TW and the total race time Trace. For the sake of space, the
model equations are not presented in this paper, but all the
calculations happen in the blocks Vehicle’s states and Race
time.
Given the current inputs and states of the system, the lap
time is computed using lap time maps. For instance, allocating more battery energy or using a fresh soft compound
results in faster lap times. On the other hand, deteriorated
tires, a heavy car, or a pit stop increase the lap time. To
account for the interaction between agents, we extend the
lap time computation of [10] as
Tlap =Tnom(Eb, ∆Eb,all, ∆Ef,all, mcar,PS)
+ ∆Tj (TW) + ∆Tint, (2)
Race car
Lap time
Vehicle’s states
Race time
a ∆Tint, tgap
∆Eb,all ∆Ef,all PS
Eb mcar TW TC
∆Tlap
tgap
Tlap
Trace
o ˜o
Fig. 2. Schematic of the Race car model. Inputs are the agent’s action
a, the gap time to the opponent tgap and the additional lap time caused
by the aerodynamic interaction ∆Tint. Output are the observation of the
ego car o and the available observations for the opponent ˜o. For a detailed
mathematical description, the reader is referred to [10].
where Tnom is the nominal lap time map, ∆Tj is the
additional time given by the tire wear and the chosen
compound j, and ∆Tint is the additional lap time caused
by the interaction.
Following the formulation of [10], the resulting states of
the car are
s =

Eb Ef mcar Trace . . .
bcpd TC TW boutlap
, (3)
where bcpd and boutlap are auxiliary variables indicating if
at least two different compounds were employed (according
to regulations) and if the previous lap was an outlap, respectively. These variables are needed to recover the Markov
property [10].
We distinguish between the observations of the ego car o
and of the competitor ˜o. From the competitor’s perspective,
not all the states of the other car are completely observable.
For instance, the battery level and its allocation strategy are
not available information to other teams. Nevertheless, they
can observe quantities such as the tire compound, the gap
time, or if a pit stop is called. They can also track the tire
age TA or if at least two different compounds were employed
to meet the regulations. We choose the observation vector for
the ego car as
o =

s Tlap Nlaps − k

, (4)
to inform the agent about the resulting lap time and the
number of remaining laps. The observations available to the
competitor are
˜o =

TA PS bcpd tgap
. (5)
Note that we choose the gap time tgap instead of the lap
time, although the latter is publicly available. This choice is
motivated by two considerations. First, given the ego lap time
and the gap time, the competitor’s lap time is a redundant
information. Second, the gap time is a derived variable that
directly correlates with the additional lap time induced by
the interaction. This facilitates the learning of the physical
coupling.
C. Aerodynamic interaction model
This model captures the physical coupling between agents.
Slipstreaming reduces drag and downforce, accelerates tire
degradation, and deteriorates brakes and engine cooling.
These effects affect the lap time and influence the race
strategy. In this work, only drag and downforce reduction
are considered, although the framework can be extended to
include additional interaction effects.
The dominance of drag or downforce depends on the
characteristics of the circuit. On high-speed circuits such as
the Autodromo Nazionale di Monza, drag reduction benefits
the trailing car and can decrease lap times. Conversely, the
Bahrain International Circuit features a lot of corners, and
the downforce loss outweighs the drag reduction, resulting
in slower laps.
A numerical investigation is possible with the gametheoretic framework presented in [15]. For an initial tgap,
the difference in lap time ∆Tint is computed. The resulting
trend is fitted, and for the Bahrain International Circuit we
model it as
∆Tint =
(
a · tgap + b, if tgap ∈ [0.2 s, . . . , 1.5 s],
0 else,
(6)
where a < 0 and b are fitting coefficients, equal for both
cars. Where the function is non-zero, the values of a and b
result in a lap time loss, i.e., being behind is detrimental in
terms of lap time.
Finally, the gap times dynamics are modeled as
t
1
gap[k + 1] = t
1
gap[k] + (T
1
lap[k] − T
i
lap[k]), (7)
t
i
gap[k + 1] = t
i
gap[k] + (T
i
lap[k] − T
1
lap[k]), (8)
where tgap > 0 indicates that the agent is behind.
D. Agent’s architecture
We introduce an agent structure to effectively handle the
competitor’s information, shown in Figure 3. The pre-trained
single-agent policy of [10] is used as a backbone, whose
neural network’s layers are kept frozen during training. Its
output is the nominal action anom. Inspired by [16] and by
Deepmind’s α-fold modular structure [17], we extend the
agent with an interaction module, which outputs the action
correction ∆a. The final policy is obtained by combining the
outputs of the two modules as
a = anom + ∆a. (9)
The single-agent policy takes as input the observation vector
o, containing the complete state information of the ego
vehicle. To compute the correction term, the interaction
module receives the same ego observation o and, in addition,
the opponent’s information ˜o.
Agent
Single-agent
policy
Interaction
module
+
o ˜o
anom ∆a
a
Fig. 3. Schematic of the agent’s structure. The single-agent policy is taken
from [10] and its weights are kept frozen during training, while only the
interaction module is trained. Inputs are the observation of the ego car o
and the ones about the opponent ˜o. The nominal policy anom is combined
with the policy of the interaction module ∆a to output the action a.
This architecture has two main advantages. First, the nominal policy already provides a robust baseline for single-agent
race strategies, as shown in [10]. Second, the computational
burden associated with training agents de novo is mitigated,
stabilizing the training. The extended agent only requires
fine-tuning to account for interaction effects, because only
the correction from the nominal policy has to be learned.
E. Markov decision process
The system is described by the state space
S =

s
1
t
1
gap s
i
t
i
gap
∈ R
18
, (10)
the observation space
O =

o
1 ˜o
i

∈ R
14
, (11)
and the action space
A =

a
1

∈ R
3
, (12)
such that S, O, A are feasible in the environment. The
deterministic transition function
T : S × A → S such that s
′
k+1 = T(s
′
k
, a
′
k
), (13)
with s
′ ∈ S and a
′ ∈ A, captures the transition dynamics
from lap k to lap k + 1. The episode terminates when no
laps are left, i.e., k = Nlaps.
The objective is to finish the race ahead of the opponent.
However, a pure winner reward encourages detrimental behaviors, with agents focusing primarily on interfering with
one another. To promote reasonable and realistic strategies
that account for interaction effects, we adopt the reward
function
R(s
′
k
, ak, s
′
k+1) = rk
= rstep + rfinal. (14)
It is composed by
rstep = Tc − T
1
lap[k], ∀k ∈ [0, . . . , Nlaps], (15)
where Tc is a constant offset that keeps the reward numerically well-scaled, and
rfinal =
(
cwin, if t
1
gap[Nlaps] < 0,
0, else,
(16)
where cwin is a constant winner reward, chosen to be approximately one order of magnitude smaller than the cumulative
reward. Maximizing the cumulative reward therefore corresponds primarily to minimizing the race time, with winning
as a secondary objective. The discount factor is set to γ = 1.
Since the process satisfies the Markov property, by definition the transition probability satisfies
P

sk+1 | s0, . . . , sk, a0, . . . , ak

= P

sk+1 | sk, ak

, (17)
for all k. Finally, the finite-horizon MDP
M = (S, O, A, T, R, P, γ) (18)
formalizes the race strategy optimization problem considering the competitor’s interaction.
III. MULTI-AGENT EXTENSION
In this section, we describe the generation of multiple
agents for the battle arena. A self-play training scheme is
employed, and a ranking system is introduced to assess and
order agents’ performance.
A. Self-play training
Training methods for multi-agent systems are computationally demanding. Common approaches include centralized
and decentralized multi-agent RL. The former is well suited
for cooperative tasks, as agents share their experiences,
whereas the latter is more appropriate for competitive settings, where agents learn independently.
Self-play [18], [19] provides an efficient and effective
framework for training structurally identical agents. One
agent interacts with the environment while competing against
an identical, fixed opponent embedded in it. During training,
only the learning agent is updated. After convergence, the
opponent is replaced with the newly trained agent, and a
new self-play iteration begins. The policy is progressively
improved by competing always against an opponent with
similar expertise.
In the environment shown in Figure 1, Agent 1 is the
learning agent, whereas Agent i is the opponent updated at
each self-play iteration. Although this procedure breaks the
Markov property across iterations, it still holds within each
training episode.
Combining self-play with the interaction module improves
learning efficiency. First, training is performed on a single
agent at a time, avoiding the overhead associated with
decentralized training of multiple agents. Second, only the
interaction module is updated, refining the already trained
single-agent policy.
Agent1,1
Agent1,best
Iter. 1
Agentn,1
Agentn,best
Iter. n
Environment
Single-agent
policy
Opponent
Agent1,best
.
.
.
Agentn−1,best
Training
a
o
.
.
.
Training
a
o
Fig. 4. Custom self-play training scheme. The training agent is shown on
the left, while the opponent embedded in the environment remains fixed.
During the first iteration, the single-agent policy is the only opponent. Then,
the training agent with the highest Elo score (“best”), is selected for the pool
of future opponents of iteration n.
Custom self-play: To diversify training and encourage
exploration, we modify the basic self-play scheme, as illustrated in Figure 4. During the first self-play iteration, the
opponent is restricted to the single-agent policy, preventing
the learning of strategies based on an uninitialized interaction
module. Moreover, instead of waiting until convergence,
the opponent embedded in the environment is intermittently
selected at random from a pool of previously trained agents.
Training details: Before each episode, the initial gap time
is randomly sampled to promote robustness. A soft actorcritic (SAC) algorithm is used and the training of the agent
on a commercial laptop (Apple M2 Max, 32 GB RAM) takes
approximately 3 h.
B. Ranking system
Agents with similar rewards may have different performance, because the reward function balances cumulative race
time with winner reward. Thus, an agent with suboptimal
pace may still win under favorable initial conditions, while
a faster agent facing adverse conditions may lose.
We adopt the Elo rating system [20] to measure the
policies’ relative performance. After each match, ratings
are updated based on the difference between the players’
rankings: Defeating a stronger opponent yields a larger
increase than defeating a weaker one. This mechanism is
more representative of the relative performance than the
winning probability, since it prevents agents from achieving
high rankings only by outperforming weak opponents.
The Elo rating can be continuously updated. During training, we select previous opponents based on their ranking
to generate the pool of agents for the self-play iteration.
After training, high-ranked agents compete until their ratings
Agent A B C D
Elo score 1970 946 914 970
Ranking 1
◦ 3
◦ 4
◦ 2
◦
TABLE II
ELO SCORE AND RANKING POSITION FOR THE AGENTS OF THE
CONSIDERED BATTLE ARENA.
converge, creating a battle arena that orders agents by
performance and allows new ones to be integrated later.
IV. RESULTS
In this section, we showcase the validity of the presented
framework. In a first case study, we analyze a race between
two agents. In the second case study, we show how the agents
adapt their strategy to responding opponents.
Table II summarizes the Elo score and ranking position for
the four agents constituting the considered battle arena. To
define the pit stop strategy we use the notation (TCk, . . .),
where TC ∈ {S, M, H} is the tire compound (S stands for
“soft”, M for “medium” and H for “hard”), and k is the lap
where that compound is mounted.
The system has to be initialized. To allow for fair comparison, every agent starts on medium tires and agent A always
starts the race 0.5 s behind the other agents, simulating a
realistic race start.
A. Battle between A and B
Figure 5 shows the fuel energy allocation, the battery
energy allocation, the pit stops decisions, and the race time
difference for A against B. In particular, A chooses a
(M0, S19, S33) strategy while B goes for (M0, S22, S49).
For the first third of the race, A remains behind B. Until
the first pit stop, A maintains a gap of approximately 1.6 s,
staying outside the wake effect region that would otherwise
increase the lap time. The fuel energy allocation strategy
substantially differs, with A allocating less fuel per lap than
B.
Thanks to the first pit stop phase, A performs a successful
undercut. After pitting on lap 19, A rejoins the track outside
the wake of B. With fresh soft tires and exploiting the
previously conserved fuel, A reduces the gap. Indeed, when
B pits at lap 22, it rejoins behind.
Just before being overtaken, A pits again at lap 33,
avoiding the disadvantage of the aerodynamic interaction.
Thanks to fresher tires and the lighter car, A closes the gap to
B over the subsequent 17 laps, while B increasingly suffers
from tire degradation effects. At this point, A secured the
race lead, as it will remain ahead regardless of B’s strategy.
Nevertheless, B decides to pit anyway due to severe tire wear
(soft tires with an age of 27 laps), prioritizing a competitive
race time.
A wins the race by 12.52 s. Balancing tire degradation
with fuel allocation, A prevails over B over a race lasting
∆
Ef,all [−]
∆
Eb,all [−]
Lap
∆
T [s]
90%
100%
110%
−1
0
1
No pit
Pit for S
Pit for M
Pit for H
0 10 20 30 40 50
−30
−20
−10
0
10
20
30
Agent A
Agent B
∆T < 0 → A is ahead.
Fig. 5. Race strategies and race time difference for the duel between A
(in blue) and B (in red). The first plot shows the normalized fuel energy
allocation, the second one the normalized battery energy allocation, and the
third one the pit stop decision variable. Agent A starts 0.5 s behind B, and
a negative gap time means that agent A is ahead.
approximately 90 min. We observe how both agents react to
the current race state, responding to the competitor’s actions
and observing its behavior.
B. Adapting the strategy
To further illustrate the agent response to different opponents, we evaluate pairwise races in which two agents
compete simultaneously. The resulting pit stop strategies
are shown in Figure 6. Race times are compared against
a constant baseline defined as the race time of A when
competing against C
∆Trace = Trace,i(Nlaps) − TBaseline, (19)
and are displayed on the right y-axis of Figure 6.
Agent A wins against every opponent. While B, C, and
D have comparable Elo scores, the difference of 1000 points
indicates a clear dominance of A, even with different initial
gap times (not shown). It is important to emphasize that
energy management constitutes a critical component of race
strategy, and race outcomes are not only determined by pitstop decisions.
The policy of A is distinguished by its robustness.
Across all duels, A consistently adopts a two-stop strategy
(M, S, S), modifying the pit-stop laps to react to the opponent. This behavior indicates that the interaction module
works as intended. Indeed, variations in the pit stop timing
are performed to react to the competitor’s response, while
preserving the underlying strategy. The energy management
Lap
∆Trace
0 5 10 15 20 25 30 35 40 45 50 55
B
C
D
A
C
A
B
A
+11.36 s
+19.26 s
+8.80 s
+2.48 s
+19.95 s
Baseline
+14.66 s
+1.84 s
Fig. 6. Pit stop and tire compound strategies for duels between different
agents. Yellow indicates the medium compound and red the soft compound.
Tire icons mark the pit stop laps. In each pair, the agent shown above starts
0.5 s behind the agent below. On the right side, the race time differences are
shown, all computed with respect to a common baseline, defined by agent
A competing against C.
is adjusted accordingly, although omitted here for brevity.
Although B, C, and D explore alternative strategies, they
do not result in successful ones.
All agents are consistent in their own race times, even
when using different strategies. Regardless of the opponent,
the ∆Trace of A remains on the same order of magnitude.
Similar consistency is observed for B and C, whose race
time vary only marginally when racing against different
opponents. Moreover, race times correlates with the agents’
ranking, with higher-ranked agents consistently achieving
lower total race times.
Except for A, the other agents change their strategies
depending on the opponent. For instance, when C competes
against A and starts ahead, it adopts a three-stops strategy,
whereas when starting behind B it switches to a two-stops
strategy. Similarly, B performs two stops when starting ahead
of A, but reduces to a single stop when racing against C.
From a race strategist perspective, selecting among multiple
strategies with comparable race times remains challenging.
This further motivates the use of RL agents to identify the
most robust choice.
In general, all the agents show a clear preference for the
soft compound. While this choice aligns with common practice at the Bahrain International Circuit, it strongly depends
on the tire degradation model and on the pre-trained singleagent policy. Diversification of training on initial compounds
and further investigation of the self-play training could reveal
new strategies.
V. CONCLUSIONS
In this paper, we presented a framework for training RL
agents that determines F1 race strategies while accounting for
interactions with competitors. Each agent decides the energy
management, pit-stop timing, and tire compound selection.
Building upon a single-agent policy, we extended the agent
by introducing an interaction module to react to opponents’
behavior. The combination of this module with a self-play
training scheme is crucial for efficient learning of multi-agent
settings. Moreover, the agent’s backbone pre-trained policy
retains the robust race time performance.
The proposed approach generates agents with distinct
performance levels and strategic behaviors. Results show
that they adapt their strategy based on the current race state
and competitors’ actions. As the agents rely exclusively on
information available during real races, this tool can support
the decision-making of race strategists, complementing the
large number of Monte Carlo simulations typically required.
Future work could incorporate the inclusion of traffic
through probabilistic representations, as well as additional
stochastic factors such as safety cars or weather predictions.
The learned policies could also be analyzed to identify
potential Nash equilibria in strategic interactions. Concerning
training, opponents’ labeling may improve the performance,
by allowing the interaction module to tailor responses to
specific competitors.
REFERENCES
[1] A. Heilmeier, A. Thomaser, M. Graf, and J. Betz, “Virtual strategy
engineer: Using artificial neural networks for making race strategy
decisions in circuit motorsport,” Applied Sciences, vol. 10, no. 21, p.
7805, 2020.
[2] P. Duhr, D. Buccheri, C. Balerna, A. Cerofolini, and C. H. Onder, “Minimum-race-time energy allocation strategies for the hybridelectric Formula 1 power unit,” IEEE Transactions on Vehicular
Technology, vol. 72, no. 6, pp. 7035–7050, 2023.
[3] J. van Kampen, T. Herrmann, and M. Salazar, “Maximum-distance
race strategies for a fully electric endurance race car,” European
Journal of Control, vol. 68, p. 100679, 2022.
[4] A. Bonomi, E. Turri, and G. Iacca, “Evolutionary F1 race strategy,”
in Proceedings of the Companion Conference on Genetic and Evolutionary Computation, 2023, pp. 1925–1932.
[5] O. F. C. Heine and C. Thraves, “On the optimization of pit stop
strategies via dynamic programming,” Central European Journal of
Operations Research, vol. 31, no. 1, pp. 239–268, 2023.
[6] M.-P. Neumann, G. Fieni, F. Furia, A. Cerofolini, V. Ravaglioli, C. H.
Onder, and G. Zardini, “Strategic co-design in Formula 1: Balancing
physical configuration and race tactics,” 2024.
[7] M. Boettinger and D. Klotz, “Mastering Nordschleife–A comprehensive race simulation for AI strategy decision-making in motorsports,”
arXiv preprint arXiv:2306.16088, 2023.
[8] D. Thomas, J. Jiang, A. Kori, A. Russo, S. Winkler, S. Sale,
J. McMillan, F. Belardinelli, and A. Rago, “Explainable reinforcement
learning for Formula One race strategy,” in Proceedings of the 40th
ACM/SIGAPP Symposium on Applied Computing, 2025, pp. 1090–
1097.
[9] X. Liu, A. Fotouhi, and D. J. Auger, “Formula-E race strategy
development using distributed policy gradient reinforcement learning,”
Knowledge-Based Systems, vol. 216, p. 106781, 2021.
[10] G. Fieni, J. Wuthrich, M.-P. Neumann, M. M. Moradi, and C. H. ¨
Onder, “Towards learning-based Formula 1 race strategies,” arXiv
preprint arXiv:2512.21570, 2025.
[11] L. Paparusso, M. Riani, F. Ruggeri, and F. Braghin, “Competitorsaware stochastic lap strategy optimisation for race hybrid vehicles,”
IEEE Transactions on Vehicular Technology, vol. 72, no. 3, pp. 3074–
3089, 2022.
[12] J. van Kampen, M. Moriggi, F. Braghin, and M. Salazar, “Model
predictive control strategies for electric endurance race cars accounting
for competitors’ interactions,” IEEE Control Systems Letters, vol. 8,
pp. 1799–1804, 2024.
[13] F. Aguad and C. Thraves, “Optimizing pit stop strategies in Formula
1 with dynamic programming and game theory,” European Journal of
Operational Research, vol. 319, no. 3, pp. 908–919, 2024.
[14] X. Liu, A. Fotouhi, and D. Auger, “Formula-E multi-car race strategy
development—a novel approach using reinforcement learning,” IEEE
Transactions on Intelligent Transportation Systems, vol. 25, no. 8, pp.
9524–9534, 2024.
[15] G. Fieni, M.-P. Neumann, F. Furia, A. Caucino, A. Cerofolini,
V. Ravaglioli, and C. H. Onder, “Game theory in Formula 1: From
physical to strategic interactions,” arXiv preprint arXiv:2503.05421,
2025.
[16] A. Mohseni-Kabir, D. Isele, and K. Fujimura, “Interaction-aware
multi-agent reinforcement learning for mobile agents with individual
goals,” in 2019 International Conference on Robotics and Automation
(ICRA). IEEE, 2019, pp. 3370–3376.
[17] J. Jumper, R. Evans, A. Pritzel, T. Green, M. Figurnov, O. Ronneberger, K. Tunyasuvunakool, R. Bates, A. Zˇ´ıdek, A. Potapenko
et al., “Highly accurate protein structure prediction with alphafold,”
nature, vol. 596, no. 7873, pp. 583–589, 2021.
[18] D. Silver, J. Schrittwieser, K. Simonyan, I. Antonoglou, A. Huang,
A. Guez, T. Hubert, L. Baker, M. Lai, A. Bolton et al., “Mastering
the game of go without human knowledge,” nature, vol. 550, no. 7676,
pp. 354–359, 2017.
[19] D. Silver, T. Hubert, J. Schrittwieser, I. Antonoglou, M. Lai, A. Guez,
M. Lanctot, L. Sifre, D. Kumaran, T. Graepel et al., “A general
reinforcement learning algorithm that masters chess, shogi, and go
through self-play,” Science, vol. 362, no. 6419, pp. 1140–1144, 2018.
[20] A. E. Elo, “The rating of chessplayers, past and present,” (No Title),