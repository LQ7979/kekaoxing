function out = ress2026_reliability_milp_demo(whichModel)
% Reproduction-style MILP for:
% - Model 2: fully-switched, radially-operated meshed networks (Eq. 7-15)
% - Model 4: partially-switched, radially-operated meshed networks (Eq. 25-28 + 7-9)
%
% Outputs:
%   out.pi, out.Gamma (per bus), out.SAIFI, out.SAIDI, out.ASAI, out.EENS_MWh
%
% Requirement: Optimization Toolbox (intlinprog)

if nargin < 1, whichModel = "model2"; end
whichModel = lower(string(whichModel));

% ---------------------------
% 1) Input data (replace with your test system)
% ---------------------------
net = makeToy24();      % <- 用你论文算例替换这里（24/54/136/... 或 37/85/...）
opts = struct();
opts.switchingTimeHr = 0.1;   % τ^SO (小时). 论文若有数值请替换
opts.bigM = [];               % 留空则自动估计
opts.wSAIFI = 1;              % 目标函数(32)的权重（可按文中设置）
opts.wSAIDI = 1;
opts.wEENS  = 1;

% ---------------------------
% 2) Solve
% ---------------------------
switch whichModel
    case "model2"
        sol = solve_model2_milp(net, opts);
    case "model4"
        sol = solve_model4_milp(net, opts);
    otherwise
        error("whichModel must be 'model2' or 'model4'.");
end

% ---------------------------
% 3) Compute indices (Eq. 1)
% ---------------------------
out = compute_indices(net, sol.pi, sol.Gamma);
out.solution = sol;

disp("=== System indices ===");
disp(struct2table(rmfield(out, "solution")));

end

% ======================================================================
% Data format
% ======================================================================
function net = makeToy24()
% 你需要替换成论文的测试系统数据：
% - buses: customers N_i, loads P_{i,b} 与 δ_b
% - edges: (u,v), λ_ij (次/年), τ^RS_ij (小时), S_ij(是否有开关)
%
% 这里给一个小的“像网状但要径向运行”的示例。

nb = 10;
sub = 1;                       % Ω^S
dist = setdiff(1:nb, sub);     % Ω^D

customers = ones(nb,1);
customers(sub) = 0;

% 单一负荷水平 b=1: δ=8760 小时/年
delta = 8760;
Pkw = zeros(nb,1);
Pkw(dist) = [120;80;60;40;30;50;70;55;35];

% Edges (undirected candidate lines)
% u v lambda repairHr hasSwitch
E = [
 1 2  0.20  5.0  1
 2 3  0.15  5.0  1
 3 4  0.10  5.0  1
 4 5  0.12  5.0  1
 5 6  0.18  5.0  1
 3 7  0.14  5.0  1
 7 8  0.10  5.0  1
 8 9  0.11  5.0  0   % 没有开关（Model4 用得到）
 9 10 0.09  5.0  1
 6 10 0.08  5.0  1   % tie，形成网状
];

edges.u = E(:,1);
edges.v = E(:,2);
edges.lambda = E(:,3);     % λ_ij (次/年)
edges.repairHr = E(:,4);   % τ^RS_ij (小时)
edges.hasSwitch = logical(E(:,5)); % S_ij (1 有开关, 0 无开关)

net.nb = nb;
net.substations = sub;
net.distNodes = dist;
net.customers = customers;
net.delta = delta;
net.Pkw = Pkw;
net.edges = edges;
end

% ======================================================================
% Common helpers
% ======================================================================
function g = build_arcs(net)
% Build directed arcs for each undirected edge.
u = net.edges.u(:);
v = net.edges.v(:);
nl = numel(u);

from = [u; v];
to   = [v; u];
arc2edge = [(1:nl)'; (1:nl)'];

g.nl = nl;
g.m = 2*nl;
g.from = from;
g.to = to;
g.arc2edge = arc2edge;
g.lambdaArc = net.edges.lambda(arc2edge);
g.repairArc = net.edges.repairHr(arc2edge);
g.hasSwitchEdge = net.edges.hasSwitch(:);

% Precompute outgoing arcs from a node, and for each arc a: downstream arcs from its head (exclude back-edge)
m = g.m;
outByNode = cell(net.nb,1);
for a = 1:m
    outByNode{from(a)} = [outByNode{from(a)}, a]; %#ok<AGROW>
end
downstream = cell(m,1);
for a = 1:m
    j = to(a);
    i = from(a);
    cand = outByNode{j};
    % exclude arc j->i if exists
    keep = true(size(cand));
    for t = 1:numel(cand)
        aa = cand(t);
        if g.to(aa) == i
            keep(t) = false;
        end
    end
    downstream{a} = cand(keep);
end
g.downstream = downstream;
end

function M = estimate_bigM(net, opts)
sumLam = sum(net.edges.lambda);
maxRepair = max(net.edges.repairHr);
ts = opts.switchingTimeHr;
% big-M for rates and durations aggregations
M = 10 * (sumLam * (maxRepair + ts) + 1);
end

function out = compute_indices(net, pi, Gamma)
dist = net.distNodes(:);
N = net.customers(:);
P = net.Pkw(:);
delta = net.delta;

totalN = sum(N(dist));
SAIFI = sum(N(dist) .* pi(dist)) / max(totalN,1);
SAIDI = sum(N(dist) .* Gamma(dist)) / max(totalN,1);
ASAI  = 1 - SAIDI/8760;

% EENS (Eq. 1d). 单负荷水平：EENS = delta * Σ P_i * Γ_i / 8760
EENS_kWh = delta * sum(P(dist) .* Gamma(dist)) / 8760;
EENS_MWh = EENS_kWh / 1000;

out.pi = pi(:);
out.Gamma = Gamma(:);
out.SAIFI = SAIFI;
out.SAIDI = SAIDI;
out.ASAI  = ASAI;
out.EENS_MWh = EENS_MWh;
end

% ======================================================================
% Model 2 MILP (Fully-switched, radially-operated meshed)
% Implements: (7)-(9), (10)-(15), and objective similar to (32).
% ======================================================================
function sol = solve_model2_milp(net, opts)
g = build_arcs(net);

nb = net.nb;
nl = g.nl;
m  = g.m;

subs = net.substations(:);
dist = net.distNodes(:);

ts = opts.switchingTimeHr;
if isempty(opts.bigM), M = estimate_bigM(net, opts); else, M = opts.bigM; end

% Decision variables
psi  = optimvar('psi', m,  'Type','integer','LowerBound',0,'UpperBound',1); % ψ_ij
y    = optimvar('y',   nl, 'Type','integer','LowerBound',0,'UpperBound',1); % y_ij
alphaSO = optimvar('alphaSO', m, 'LowerBound',0); % α^SO_ij
betaSO  = optimvar('betaSO',  m, 'LowerBound',0); % β^SO_ij
piVar   = optimvar('pi',    nb, 'LowerBound',0);  % π_i
GVar    = optimvar('Gamma', nb, 'LowerBound',0);  % Γ_i

prob = optimproblem('ObjectiveSense','minimize');

cons = optimconstr(0,1);
k = 0;

% (7) each distribution node has exactly one incoming arc active
for ii = 1:numel(dist)
    i = dist(ii);
    inArcs = find(g.to == i);
    k=k+1; cons(k) = sum(psi(inArcs)) == 1;
end

% (8) substations have no incoming supply
for ss = 1:numel(subs)
    s = subs(ss);
    inArcs = find(g.to == s);
    k=k+1; cons(k) = sum(psi(inArcs)) == 0;
end

% (9) psi_ij + psi_ji = y_edge
for e = 1:nl
    a_uv = e;        % u->v
    a_vu = e + nl;   % v->u
    k=k+1; cons(k) = psi(a_uv) + psi(a_vu) == y(e);
end

% (10) alphaSO recursion with big-M, only active when psi(a)=1
for a = 1:m
    lam = g.lambdaArc(a);
    ds = g.downstream{a};
    k=k+1; cons(k) = alphaSO(a) >= lam + sum(alphaSO(ds)) - M*(1-psi(a)); % (10a)
    k=k+1; cons(k) = alphaSO(a) <= lam + sum(alphaSO(ds)) + M*(1-psi(a)); % (10b)
    k=k+1; cons(k) = alphaSO(a) <= M*psi(a);                               % (10c)
end

% (11) root ENIF π_s for each substation s: enforce on each outgoing arc s->j
for ss = 1:numel(subs)
    s = subs(ss);
    outArcs = find(g.from == s);
    for t = 1:numel(outArcs)
        a = outArcs(t);
        lam = g.lambdaArc(a);
        ds = g.downstream{a};
        k=k+1; cons(k) = piVar(s) >= lam + sum(alphaSO(ds)) - M*(1-psi(a)); % (11a)
        k=k+1; cons(k) = piVar(s) <= lam + sum(alphaSO(ds)) + M*(1-psi(a)); % (11b)
    end
end

% (12) propagate ENIF equality: if psi(i->j)=1 then pi(j)=pi(i)
for a = 1:m
    i = g.from(a); j = g.to(a);
    if ismember(i, dist) && ismember(j, dist)
        k=k+1; cons(k) = piVar(j) >= piVar(i) - M*(1-psi(a)); % (12a)
        k=k+1; cons(k) = piVar(j) <= piVar(i) + M*(1-psi(a)); % (12b)
    end
end

% (13) betaSO recursion (fully-switched => switching-only per fault uses τ^SO)
for a = 1:m
    lam = g.lambdaArc(a);
    ds = g.downstream{a};
    k=k+1; cons(k) = betaSO(a) >= ts*lam + sum(betaSO(ds)) - M*(1-psi(a)); % (13a)
    k=k+1; cons(k) = betaSO(a) <= ts*lam + sum(betaSO(ds)) + M*(1-psi(a)); % (13b)
    k=k+1; cons(k) = betaSO(a) <= M*psi(a);                                % (13c)
end

% (14) root ENID Γ_s for substation s via its active outgoing arc
for ss = 1:numel(subs)
    s = subs(ss);
    outArcs = find(g.from == s);
    for t = 1:numel(outArcs)
        a = outArcs(t);
        lam = g.lambdaArc(a);
        rrs = g.repairArc(a);
        ds = g.downstream{a};
        k=k+1; cons(k) = GVar(s) >= rrs*lam + sum(betaSO(ds)) - M*(1-psi(a)); % (14a)
        k=k+1; cons(k) = GVar(s) <= rrs*lam + sum(betaSO(ds)) + M*(1-psi(a)); % (14b)
    end
end

% (15) downstream ENID recursion: Γ_j = Γ_i + Δτ_ij*λ_ij when psi(i->j)=1
% Here Δτ_ij uses repairHr (τ^RS). If you have τ^RS-τ^SO, replace below.
for a = 1:m
    i = g.from(a); j = g.to(a);
    if ismember(j, dist) && ~ismember(j, subs)
        lam = g.lambdaArc(a);
        dt = g.repairArc(a); % Δτ
        k=k+1; cons(k) = GVar(j) >= GVar(i) + dt*lam - M*(1-psi(a)); % (15a)
        k=k+1; cons(k) = GVar(j) <= GVar(i) + dt*lam + M*(1-psi(a)); % (15b)
    end
end

prob.Constraints.all = cons;

% Objective (32-like): wEENS*EENS + wSAIDI*SAIDI + wSAIFI*SAIFI
N = net.customers(:);
P = net.Pkw(:);
distMask = false(nb,1); distMask(dist)=true;
totalN = sum(N(dist));
SAIFI_expr = sum(N(distMask).*piVar(distMask)) / max(totalN,1);
SAIDI_expr = sum(N(distMask).*GVar(distMask)) / max(totalN,1);
EENS_expr  = net.delta * sum(P(distMask).*GVar(distMask)) / 8760 / 1000;

prob.Objective = opts.wSAIFI*SAIFI_expr + opts.wSAIDI*SAIDI_expr + opts.wEENS*EENS_expr;

ipopts = optimoptions('intlinprog','Display','off');
solRaw = solve(prob, 'Options', ipopts);

sol.psi = round(solRaw.psi);
sol.y   = round(solRaw.y);
sol.pi  = solRaw.pi;
sol.Gamma = solRaw.Gamma;
sol.alphaSO = solRaw.alphaSO;
sol.betaSO  = solRaw.betaSO;
end

% ======================================================================
% Model 4 MILP (Partially-switched, radially-operated meshed)
% Implements: (7)-(8), (25)-(27), and replaces (9) by (28).
% NOTE: β^SO is reused similar to Model 2; if you provide the exact paper
% constraints for β^SO under Model 4, I can update this part precisely.
% ======================================================================
function sol = solve_model4_milp(net, opts)
g = build_arcs(net);

nb = net.nb;
nl = g.nl;
m  = g.m;

subs = net.substations(:);
dist = net.distNodes(:);

ts = opts.switchingTimeHr;
if isempty(opts.bigM), M = estimate_bigM(net, opts); else, M = opts.bigM; end

% Decision variables
psi  = optimvar('psi', m,  'Type','integer','LowerBound',0,'UpperBound',1); % ψ_ij
y    = optimvar('y',   nl, 'Type','integer','LowerBound',0,'UpperBound',1); % y_ij (only meaningful when hasSwitch=1)
gammaVar = optimvar('gamma', m, 'LowerBound',0);  % γ_ij in Eq(25)
betaSO   = optimvar('betaSO',m, 'LowerBound',0);  % β^SO (used in Eq(26))
piVar    = optimvar('pi',   nb,'LowerBound',0);   % π_i (still needed for SAIFI if you optimize by it)
GVar     = optimvar('Gamma',nb,'LowerBound',0);   % Γ_i

prob = optimproblem('ObjectiveSense','minimize');
cons = optimconstr(0,1);
k = 0;

% (7) indegree = 1 for distribution nodes
for ii = 1:numel(dist)
    i = dist(ii);
    inArcs = find(g.to == i);
    k=k+1; cons(k) = sum(psi(inArcs)) == 1;
end

% (8) substations indegree = 0
for ss = 1:numel(subs)
    s = subs(ss);
    inArcs = find(g.to == s);
    k=k+1; cons(k) = sum(psi(inArcs)) == 0;
end

% (28) replaces (9):
% if hasSwitch(edge)=1: psi_uv + psi_vu = y(edge)
% if hasSwitch(edge)=0: psi_uv + psi_vu = 1  (always energized, direction chosen)
Sedge = net.edges.hasSwitch(:);
for e = 1:nl
    a_uv = e;
    a_vu = e + nl;
    if Sedge(e)
        k=k+1; cons(k) = psi(a_uv) + psi(a_vu) == y(e);
    else
        k=k+1; cons(k) = psi(a_uv) + psi(a_vu) == 1;
        k=k+1; cons(k) = y(e) == 1; % make y irrelevant but fixed
    end
end

% (25) segment interruptions gammaVar with switch presence (1-S_ij)
for a = 1:m
    e = g.arc2edge(a);
    S = double(Sedge(e));      % S_ij
    lam = g.lambdaArc(a);
    dt  = g.repairArc(a);      % Δτ_ij (use repairHr; replace if paper defines differently)
    ds  = g.downstream{a};

    expr = (1 - S) * (dt*lam + sum(gammaVar(ds)));

    k=k+1; cons(k) = gammaVar(a) >= expr - M*(1-psi(a));  % (25a)
    k=k+1; cons(k) = gammaVar(a) <= expr + M*(1-psi(a));  % (25b)
    k=k+1; cons(k) = gammaVar(a) <= M*psi(a);             % (25c)
end

% β^SO: fallback implementation (similar structure to (13), active when psi=1)
% If you can provide the paper's exact β^SO constraints for Model 4, I will patch this.
for a = 1:m
    lam = g.lambdaArc(a);
    ds = g.downstream{a};
    k=k+1; cons(k) = betaSO(a) >= ts*lam + sum(betaSO(ds)) - M*(1-psi(a));
    k=k+1; cons(k) = betaSO(a) <= ts*lam + sum(betaSO(ds)) + M*(1-psi(a));
    k=k+1; cons(k) = betaSO(a) <= M*psi(a);
end

% (26) root ENID for substation nodes
for ss = 1:numel(subs)
    s = subs(ss);
    outArcs = find(g.from == s);
    for t = 1:numel(outArcs)
        a = outArcs(t);
        lam = g.lambdaArc(a);
        rrs = g.repairArc(a);
        ds = g.downstream{a};
        k=k+1; cons(k) = GVar(s) >= rrs*lam + sum(gammaVar(ds) + betaSO(ds)) - M*(1-psi(a)); % (26a)
        k=k+1; cons(k) = GVar(s) <= rrs*lam + sum(gammaVar(ds) + betaSO(ds)) + M*(1-psi(a)); % (26b)
    end
end

% (27) downstream ENID recursion
for a = 1:m
    i = g.from(a); j = g.to(a);
    e = g.arc2edge(a);
    S = double(Sedge(e));
    lam = g.lambdaArc(a);
    dt  = g.repairArc(a);
    ds  = g.downstream{a};

    expr = GVar(i) + S*(dt*lam + sum(gammaVar(ds)));

    k=k+1; cons(k) = GVar(j) >= expr - M*(1-psi(a)); % (27a)
    k=k+1; cons(k) = GVar(j) <= expr + M*(1-psi(a)); % (27b)
end

% π_i 这部分：你若要把 SAIFI 纳入目标，就需要 π_i。
% 这里用一个“最小可用”的定义：π_i 等于其入弧的 λ + 下游 λ 累计（类似 Model2 的 αSO 思路）。
% 若你提供 Model4 对 π/ENIF 的精确约束页，我可以完全按论文实现。
alphaSO = optimvar('alphaSO', m, 'LowerBound',0);
for a = 1:m
    lam = g.lambdaArc(a);
    ds = g.downstream{a};
    k=k+1; cons(k) = alphaSO(a) >= lam + sum(alphaSO(ds)) - M*(1-psi(a));
    k=k+1; cons(k) = alphaSO(a) <= lam + sum(alphaSO(ds)) + M*(1-psi(a));
    k=k+1; cons(k) = alphaSO(a) <= M*psi(a);
end
for ss = 1:numel(subs)
    s = subs(ss);
    outArcs = find(g.from == s);
    for t = 1:numel(outArcs)
        a = outArcs(t);
        lam = g.lambdaArc(a);
        ds = g.downstream{a};
        k=k+1; cons(k) = piVar(s) >= lam + sum(alphaSO(ds)) - M*(1-psi(a));
        k=k+1; cons(k) = piVar(s) <= lam + sum(alphaSO(ds)) + M*(1-psi(a));
    end
end
for a = 1:m
    i = g.from(a); j = g.to(a);
    if ismember(i, dist) && ismember(j, dist)
        k=k+1; cons(k) = piVar(j) >= piVar(i) - M*(1-psi(a));
        k=k+1; cons(k) = piVar(j) <= piVar(i) + M*(1-psi(a));
    end
end

prob.Constraints.all = cons;

% Objective (32-like)
N = net.customers(:);
P = net.Pkw(:);
distMask = false(nb,1); distMask(dist)=true;
totalN = sum(N(dist));

SAIFI_expr = sum(N(distMask).*piVar(distMask)) / max(totalN,1);
SAIDI_expr = sum(N(distMask).*GVar(distMask)) / max(totalN,1);
EENS_expr  = net.delta * sum(P(distMask).*GVar(distMask)) / 8760 / 1000;

prob.Objective = opts.wSAIFI*SAIFI_expr + opts.wSAIDI*SAIDI_expr + opts.wEENS*EENS_expr;

ipopts = optimoptions('intlinprog','Display','off');
solRaw = solve(prob, 'Options', ipopts);

sol.psi = round(solRaw.psi);
sol.y   = round(solRaw.y);
sol.pi  = solRaw.pi;
sol.Gamma = solRaw.Gamma;
sol.gamma = solRaw.gamma;
sol.betaSO = solRaw.betaSO;
end