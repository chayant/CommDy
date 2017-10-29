import pprint

from cvxopt import solvers, sparse, spdiag, matrix, spmatrix
from itertools import groupby
from pandas import DataFrame

pp = pprint.PrettyPrinter(indent=4)

class UnionFind:
  def __init__(self):
    self.parent = {}
    self.rank = {}

  def MakeSet(self, x):
    if x not in self.parent:
      self.parent[x] = x
      self.rank[x] = 0

  def Find(self, x):
    self.MakeSet(x)
    p = self.parent[x]
    if p != x:
      leader = self.parent[x] = self.Find(p)
    else:
      leader = self.parent[x] = x
    return leader

  def Union(self, x1, x2):
    self.MakeSet(x1)
    self.MakeSet(x2)
    x1 = self.Find(x1)
    x2 = self.Find(x2)
    r1 = self.rank[x1]
    r2 = self.rank[x2]
    if r1 == r2:
      self.rank[x1] += 1
    elif r1 < r2:
      x1, x2 = x2, x1
    self.parent[x2] = x1
    return x1

  def Partition(self):
    members = {}
    for x in self.parent:
      p = self.Find(x)
      if p in members:
        members[p].append(x)
      else:
        members[p] = [x]
    return members.values()


def validate_tgi_table():
  # unique g for each (i, t).
  return

def create_linear_program(df):
  t_min = df.time.min()
  t_max = df.time.max()
  w = {}

  # Creates nodes and edges.
  all_nodes = set()
  for i, g in df.groupby(["individual"]):
    nodes = ["%s.%s"%(t, g) for (t, g) in zip(g.time, g.group)]
    nodes = ["%s.first"%nodes[0]] + nodes + [nodes[-1] + ".last"]
    all_nodes.update(nodes)
    for u, v in zip(nodes, nodes[1:]):
      edge = (u, v)
      if edge in w:
        w[edge] += 1
      else:
        w[edge] = 1
  all_nodes = sorted(all_nodes)
  all_edges = sorted(w)
  n = len(all_nodes)
  m = len(all_edges)
  node_idx = {u: i for i, u in enumerate(all_nodes)}

  # Create constraints.
  vals = []
  rows = []
  cols = []
  for i, (u, v) in enumerate(all_edges):
    vals += [ 1., 1., -1. ]
    rows += [
      node_idx[u],
      n + node_idx[v],
      2 * n + i,
    ]
    cols += [i, i, i]
  A = spmatrix(vals, rows, cols, (2 * n + m, m))
  b = matrix([1.0] * 2 * n + [0.0] * m)
  # objective function.
  c = matrix([float(-w[e]) for e in all_edges])
  #print("A: %s"%A)
  #print("b: %s"%b)
  #print("c: %s"%c)
  return c, A, b, all_nodes, all_edges

def convert_lp_solution_to_coloring(edges, sol):
  uf = UnionFind()
  timestamp_cover = {}
  for x, (n1, n2) in sorted(zip(sol['x'], edges), reverse=True):
    if n1.endswith('first') or n2.endswith('last'):
      continue
    t1, g1 = n1.split('.')
    t2, g2 = n2.split('.')
    p1 = uf.Find(n1)
    p2 = uf.Find(n2)
    if p1 in timestamp_cover:
      cover1 = timestamp_cover[p1]
    else:
      cover1 = timestamp_cover[p1] = set([t1])
    if p2 in timestamp_cover:
      cover2 = timestamp_cover[p2]
    else:
      cover2 = timestamp_cover[p2] = set([t2])
    if len(cover1.intersection(cover2)) > 0:
      #print("ignore", (t1, g1), (t2, g2), cover1, cover2)
      continue
    #print("union", (t1, g1), (t2, g2), cover1, cover2)
    p = uf.Union(n1, n2)
    timestamp_cover[p].update([t1, t2])
  # Assign colors to groups.
  tg_color = {}
  for color_idx, members in enumerate(sorted(uf.Partition(), key=lambda x: (len(x), x), reverse=True)):
    for tg in members:
      t, g = tg.split('.')
      if t not in tg_color:
        tg_color[t] = {}
      tg_color[t][g] = color_idx + 1
  return tg_color

def test1(tgi, tg_color):
  tgi = [
    ("t1", "g1", "i1"),
    ("t2", "g1", "i1"),
  ]
  df = DataFrame(data={
    'time': [t for t, g, i in tgi],
    'group': [g for t, g, i in tgi],
    'individual': [i for t, g, i in tgi],
    })
  c, A, b = create_linear_program(df)
  sol = solvers.lp(c, A, b)

def test2():
  tgi = [
    ("t1", "g1", "i1"),
    ("t1", "g1", "i2"),
    ("t1", "g1", "i3"),
    ("t1", "g0", "i4"),
    ("t2", "g1", "i1"),
    ("t2", "g1", "i2"),
    #("t2", "g2", "i3"),
    ("t2", "g0", "i4"),
    ("t3", "g1", "i1"),
    ("t3", "g1", "i2"),
    ("t3", "g1", "i3"),
    ("t3", "g0", "i4"),
  ]
  df = DataFrame(data={
    'time': [t for t, g, i in tgi],
    'group': [g for t, g, i in tgi],
    'individual': [i for t, g, i in tgi],
  })
  c, A, b, nodes, edges = create_linear_program(df)
  sol = solvers.lp(c, A, b)
  #print("solution:")
  #for e, x in zip(edges, sol['x']):
  #  print("%s %f"%(e, x))
  tg_color = convert_lp_solution_to_coloring(edges, sol)
  group_color = []
  for t, g, _ in tgi:
    group_color.append(tg_color[t, g])
  print("group coloring:")
  print(DataFrame(data={
    'time': [t for t, g, i in tgi],
    'group': [g for t, g, i in tgi],
    'individual': [i for t, g, i in tgi],
    'group_color': group_color,
  }))

  color_individuals(tgi, tg_color)

class Cost:
  def __init__(self, value=0, debug=None, color=None, previous_color=None):
    self.value = value
    self.color = color
    self.previous_color = previous_color
    if debug is None:
      self.debug = []
    elif isinstance(debug, list):
      self.debug = debug
    else:
      self.debug = [debug]
  def __add__(a, b):
    return Cost(
      value=a.value + b.value,
      debug=a.debug + b.debug,
      color=b.color if a.color is None else a.color,
      previous_color=b.previous_color if a.previous_color is None else a.previous_color,
    )
  def __lt__(a, b):
    if a.value != b.value:
      return a.value < b.value
    if a.color != b.color and a.color is not None and b.color is not None:
      return a.color < b.color
    if a.previous_color != b.previous_color and a.previous_color is not None and b.previous_color is not None:
      return a.previous_color < b.previous_color
    return False
  def __str__(self):
    return " ".join(filter(lambda x: x is not None, [
      "%d"%self.value,
      "c%s"%self.color if self.color is not None else None,
      "pc%s"%self.previous_color if self.previous_color is not None else None,
      " ".join(self.debug),
    ]))

def color_individuals(tgi, tg_color, sw=1, ab=1, vi=1, only_individual=None):
  # Index group by (individual, time)
  it_group = {}
  for t, g, i in tgi:
    it_group[i, t] = g
  
  if only_individual is None:
    individuals = sorted(set([i for t, g, i in tgi]))
  elif isinstance(only_individual, list):
    individuals = only_individual
  else:
    individuals = [only_individual]

  times = sorted(set([t for t, g, i in tgi]))
  t_group_colors = {}
  for t in times:
    t_group_colors[t] = set(tg_color[t].values())
  prev_t = {times[idx+1]: times[idx] for idx in range(len(times) - 1)}
    
  #print("tg_color %s"%tg_color)
  # Color the individuals.
  itc_min_cost = {}  # map: (i, t, c) -> min Cost
  it_color = {}
  total_min_cost = 0
  for i in individuals:
    #subrows = df[df.individual == i]

    # Find colors.
    colors = set([0])
    for t in times:
      if (i, t) not in it_group:
        continue
      g = it_group[i, t]
      colors.add(tg_color[t][g])
    #print("colors:", colors)

    # Compute the coloring cost.
    for t in times:
      #print("= time %s ="%(t))
      if (i, t) in it_group:
        g = it_group[i, t]
        gc = tg_color[t][g]
      else:
        g = None
        gc = None

      for c in colors:
        #print("== itc %s %s c%d =="%(i, t, c))
        cost = Cost(color=c)
        if t in prev_t:
          previous_costs = []
          for pc in colors:
            pcost = itc_min_cost[i, prev_t[t], pc]
            if pc != c:
              pcost += Cost(sw, "sw")
            previous_costs.append(pcost)
          min_pcost = min(previous_costs)
          cost += min_pcost
          cost.previous_color = min_pcost.color
        if c != gc:
          if gc != None:
            cost += Cost(vi, "vi")
          if c in t_group_colors[t]:
            cost += Cost(ab, "ab")
        itc_min_cost[i, t, c] = cost
        #print("min cost", cost)
    # Trace back.
    #print("trace back")
    min_color = None
    for t in reversed(times):
      if min_color is None:
        min_cost = min(itc_min_cost[i, times[-1], c] for c in colors)
        total_min_cost += min_cost.value
      else:
        min_cost = itc_min_cost[i, t, min_color]
      #print("%s cost %s"%(t, min_cost))
      it_color[i, t] = min_cost.color
      min_color = min_cost.previous_color
    #print("individual %s"%i)
    #print(" ".join(str(c) for c in colors))
    #for t in times:
    #  print("gc%s"%tg_color[t][it_group[i, t]], ", ".join("cost %s"%itc_min_cost[i, t, c] for c in colors))
    #print()
  for i in individuals:
    print(i, ":", " ".join(str(it_color[i, t]) for t in times))
  return total_min_cost, it_color

def test_color_individuals():
  test_cases = {
      "equal costs": {
        'sw': 1, 'ab': 1, 'vi': 1,
        'tgi': [
          ('t1', 'g1', 'i1'),
          ('t1', 'g2', 'i2'),
          ('t2', 'g1', 'i1'),
          ('t2', 'g2', 'i2'),
        ],
        'tg_color': {
          't1': { 'g1': 1, 'g2': 2 },
          't2': { 'g1': 1, 'g2': 2 },
        }
      },
      "cheap vi": {
        'sw': 10, 'ab': 10, 'vi': 1,
        'tgi': [
          ('t1', 'g1', 'i1'),
          ('t1', 'g1', 'i2'),
          ('t2', 'g1', 'i1'),
          ('t2', 'g2', 'i2'),
          ('t3', 'g1', 'i1'),
          ('t3', 'g1', 'i2'),
        ],
        'tg_color': {
          't1': { 'g1': 1, 'g2': 2 },
          't2': { 'g1': 1, 'g2': 2 },
          't3': { 'g1': 1, 'g2': 2 },
        }
      },
      "cheap ab/vi": {
        'sw': 10, 'ab': 1, 'vi': 1,
        'tgi': [
          ('t1', 'g1', 'i1'),
          ('t1', 'g1', 'i2'),
          ('t2', 'g1', 'i1'),
          ('t2', 'g2', 'i2'),
          ('t3', 'g1', 'i1'),
          ('t3', 'g1', 'i2'),
        ],
        'tg_color': {
          't1': { 'g1': 1, 'g2': 2 },
          't2': { 'g1': 1, 'g2': 2 },
          't3': { 'g1': 1, 'g2': 2 },
        }
      },
      "cheap sw": {
        'sw': 1, 'ab': 10, 'vi': 10,
        'tgi': [
          ('t1', 'g1', 'i1'),
          ('t2', 'g1', 'i1'),
          ('t3', 'g1', 'i1'),
        ],
        'tg_color': {
          't1': { 'g1': 1 },
          't2': { 'g1': 2 },
          't3': { 'g1': 1 },
        }
      },
      "cheap ab/vi": {
        'sw': 10, 'ab': 1, 'vi': 1,
        'only_individual': 'i1',
        'tgi': [
          ('t1', 'g1', 'i1'),
          ('t1', 'g2', 'i2'),
          ('t1', 'g3', 'i3'),
          ('t2', 'g1', 'i1'),
          ('t2', 'g2', 'i2'),
          ('t2', 'g3', 'i3'),
          ('t3', 'g1', 'i1'),
          ('t3', 'g2', 'i2'),
          ('t3', 'g3', 'i3'),
        ],
        'tg_color': {
          't1': { 'g1': 1, 'g2': 2, 'g3': 3 },
          't2': { 'g1': 2, 'g2': 3, 'g3': 1 },
          't3': { 'g1': 3, 'g2': 1, 'g3': 2 },
        }
      },
  }
  for desc, tc in test_cases.items():
    print("Case:", desc)
    pp.pprint(tc)
    cost, it_color = color_individuals(**tc)
    print("cost:", cost)
    print()
# Test.
if __name__ == '__main__':
  test_color_individuals()
