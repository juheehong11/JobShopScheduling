from parse import read_input_file, write_output_file
import os
from ortools.algorithms import pywrapknapsack_solver # https://developers.google.com/optimization/bin/knapsack

def solve(tasks): 
    """
    Args:
        tasks: list[Task], list of igloos to polish 
    Returns:
        output: list of igloos in order of polishing  
    """
    best_result = None
    best_profit = 0
    n = len(tasks)
    duration = []
    deadline = []
    profit = []

    skipknap = False
    #analyze for duplicates/ same properties
    for i in range(n):
      duration.append(tasks[i].get_duration())
      deadline.append(tasks[i].get_deadline())
      profit.append(tasks[i].get_max_benefit())
        

    if int(sum(profit)/len(profit)) == int(profit[1]) == int(profit[50]):
        skipknap = True
    deadprof = {}     #deadline profit chunks:
    durprof = {}     #deadline profit chunks:
    deaddurprof = {}
    for i in range(n):
        if (deadline[i], profit[i]) not in deadprof:
            deadprof[(deadline[i], profit[i])] = 1
        else:
            deadprof[(deadline[i], profit[i])] += 1
        
        if (deadline[i], duration[i], profit[i]) not in deaddurprof:
            deaddurprof[(deadline[i], duration[i], profit[i])] = 1
        else:
            deaddurprof[(deadline[i], duration[i], profit[i])] += 1

        if (duration[i], profit[i]) not in durprof:
            durprof[(duration[i], profit[i])] = 1
        else:
            durprof[(duration[i], profit[i])] += 1
    for key in deadprof:
        if deadprof[key] / n >= .8:
            skipknap = True
            break
    for key in durprof:
        if durprof[key] / n >= .8:
            skipknap = True
            break
    
    for key in deaddurprof:
        if deaddurprof[key] / n >= .4:
            skipknap = False
            break
        if deaddurprof[key] > 6:
            skipknap = True
            break

    if len(durprof) < 8:
        skipknap = True
    

    if not skipknap:
        for size in range(n//8,n//2+1,n//8):
            id_ordering, profit, tasks_ordering = knapsack_solver(tasks,size)
            assert profit == get_profit(tasks_ordering)
            if profit > best_profit:
                best_profit = profit
                best_result = id_ordering
                best_ordering = tasks_ordering
    for id_ordering, profit, tasks_ordering in [greedy_profit_duration(tasks), greedy_duration_deadline(tasks), greedy_duration_deadline(tasks), greedy_critical_time(tasks),
                                                greedy_min_slack(tasks), greedy_decay(tasks), greedy_decay_profit_duration(tasks)]:
        assert profit == get_profit(tasks_ordering)
        if profit > best_profit:
            best_profit = profit
            best_result = id_ordering
            best_ordering = tasks_ordering
    
    dp_ordering = dp(tasks)
    dp_profit = get_profit(dp_ordering)
    if dp_profit > best_profit:
        best_profit = dp_profit
        best_ordering = dp_ordering
        best_result = [t.get_task_id() for t in dp_ordering]
   

    assert best_profit == get_profit(best_ordering)
    assert best_result == [t.get_task_id() for t in best_ordering]
    assert check_validity(best_result, tasks)
    return best_result

def knapsack_solver(tasks, knapsack_size, total_time=1440, recurse=1):
    solver = pywrapknapsack_solver.KnapsackSolver(
        pywrapknapsack_solver.KnapsackSolver.
        KNAPSACK_MULTIDIMENSION_BRANCH_AND_BOUND_SOLVER, 'KnapsackExample')
    #sort by deadline
    tasks_sorted_deadline = sorted(tasks, key=lambda task: task.get_deadline())
    ordering = []
    current_start = 0


    for i in range(0, len(tasks), knapsack_size):
        tasks_group = tasks_sorted_deadline[i:min(i+knapsack_size,len(tasks_sorted_deadline))]
        values = [t.get_max_benefit() for t in tasks_group]
        weights = [[t.get_duration() for t in tasks_group]]
        capacities = [min(tasks_group[-1].get_deadline(), total_time) - current_start]
        solver.Init(values, weights, capacities)
        computed_value = solver.Solve()
        packed_items = []
        for i in range(len(tasks_group)):
            if solver.BestSolutionContains(i):
                packed_items.append(tasks_group[i])
        group_ordering = sorted(packed_items, key=lambda task: task.get_deadline())
        current_start = current_start + sum([t.get_duration() for t in group_ordering])
        ordering += group_ordering
    remaining_tasks = [t for t in tasks if t not in ordering]
    time_remaining = total_time - current_start
    if remaining_tasks and recurse == 1 and time_remaining >= remaining_tasks[0].get_duration():
        second_result, second_profit, second_ordering = knapsack_solver(remaining_tasks, knapsack_size, time_remaining, 2)
        ordering += second_ordering
        time_remaining = time_remaining - sum([t.get_duration() for t in second_ordering])
        remaining_tasks = [t for t in remaining_tasks if t not in ordering]
    #after knapsack, greedily fill in remaining time
    ordering += fill_remaining(remaining_tasks, time_remaining)
    result = [t.get_task_id() for t in ordering]
    total_profit = get_profit(ordering)
    return result, total_profit, ordering

def get_profit(ordering, start_time=0):
    total_profit = 0
    current_time = start_time
    for task in ordering:
        current_time += task.get_duration()
        if current_time <= task.get_deadline():
            total_profit += task.get_max_benefit()
        else:
            total_profit += task.get_late_benefit(current_time - task.get_deadline())
    return total_profit
    
def fill_remaining(tasks_remaining, time_remaining, from_decay=False):
    # compares two different methods of filling in remaining time:
    # tries fill in backwards from 1440, latest deadline first, and tries greedy_decay
    tasks_sorted = sorted(tasks_remaining, key=lambda task: (-task.get_deadline(), -task.get_max_benefit()))
    ordering = []
    temp_time = time_remaining
    while time_remaining > 0 and tasks_sorted:
        if tasks_sorted[0].get_duration() <= time_remaining:
            ordering.append(tasks_sorted[0])
            time_remaining -= tasks_sorted[0].get_duration()
        tasks_sorted.pop(0)
    profit = get_profit(ordering, 1440 - temp_time)
    if not from_decay:
        if temp_time > 0 and tasks_remaining:
            other_id_result, other_profit, other_task_ordering = greedy_decay(tasks_remaining, temp_time)
            if other_profit > profit:
                ordering = other_task_ordering
    return ordering

def greedy_decay(tasks, time_remaining=1440):
    # different from other greedys because it allows you to go past deadline
    profit_duration = sorted(tasks, key=lambda task: (-task.get_max_benefit()/task.get_duration(), task.get_deadline()))
    profit_forwards = sorted(tasks, key=lambda task: (-task.get_max_benefit(), task.get_deadline()))
    sorted_deadline = sorted(tasks, key=lambda task: (task.get_deadline(), -task.get_max_benefit()))
    sorted_duration = sorted(tasks, key=lambda task: (task.get_duration(), task.get_deadline()))
    best_ordering = []
    best_profit = 0
    for sorted_tasks in [profit_duration, profit_forwards, sorted_deadline, sorted_duration]:
        current_time = 1440 - time_remaining
        ordering = []
        while sorted_tasks and current_time < time_remaining:
            if current_time + sorted_tasks[0].get_duration() <= time_remaining:
                ordering.append(sorted_tasks[0])
                current_time += sorted_tasks[0].get_duration()
            sorted_tasks.pop(0)
        remaining_tasks = [t for t in tasks if t not in ordering]
        reverse_ordering = ordering.copy()
        reverse_ordering.reverse()
        deadline_ordering = sorted(ordering, key=lambda task: task.get_deadline())
        for array in [deadline_ordering, ordering, reverse_ordering]:
            array += fill_remaining(remaining_tasks, time_remaining - current_time, True)
            profit = get_profit(array, 1440 - time_remaining)
            if profit > best_profit:
                best_profit = profit
                best_ordering = array
     
    result = [t.get_task_id() for t in best_ordering]
    return result, get_profit(best_ordering), best_ordering                                

def greedy_profit_duration(tasks):
    tasks_sorted = sorted(tasks, key=lambda task: task.get_max_benefit() / task.get_duration())[::-1]
    current_start = 0
    ordering = []
    for task in tasks_sorted:
        if (current_start + task.get_duration() <= task.get_deadline()):
            ordering.append(task)
            current_start += task.get_duration()
    remaining_tasks = [t for t in tasks if t not in ordering]
    ordering += fill_remaining(remaining_tasks, 1440 - current_start)
    result = [t.get_task_id() for t in ordering]
    return result, get_profit(ordering), ordering

def greedy_decay_profit_duration(tasks):
    tasks_sorted = sorted(tasks, key=lambda task: task.get_max_benefit() / task.get_duration())[::-1]

    current_start = 0
    ordering = []
    while len(tasks_sorted) > 0:
        task = tasks_sorted[0]
        if current_start + task.get_duration() <= 1440:
            ordering.append(task)
            current_start += task.get_duration()
        tasks_sorted = tasks_sorted[1:]
        tasks_sorted = sorted(tasks_sorted, key=lambda task: task.get_late_benefit(max(0, task.get_duration() - task.get_deadline())) / task.get_duration())[::-1]


    remaining_tasks = [t for t in tasks if t not in ordering]
    ordering += fill_remaining(remaining_tasks, 1440 - current_start)
    result = [t.get_task_id() for t in ordering]
    return result, get_profit(ordering), ordering

    
def greedy_duration_deadline(tasks):
    tasks_sorted = sorted(tasks, key=lambda task: (task.get_deadline(), task.get_duration())) 
    current_start = 0
    ordering = []
    for task in tasks_sorted:
        if (current_start + task.get_duration() <= task.get_deadline()):
            ordering.append(task)
            current_start += task.get_duration()
    remaining_tasks = [t for t in tasks if t not in ordering]
    ordering += fill_remaining(remaining_tasks, 1440 - current_start)
    result = [t.get_task_id() for t in ordering]
    return result, get_profit(ordering), ordering

def greedy_deadline_profit(tasks):
    tasks_sorted = sorted(tasks, key=lambda task: (task.get_deadline(), -task.get_max_benefit())) 
    current_start = 0
    ordering = []
    for task in tasks_sorted:
        if (current_start + task.get_duration() <= task.get_deadline()):
            ordering.append(task)
            current_start += task.get_duration()
    remaining_tasks = [t for t in tasks if t not in ordering]
    ordering += fill_remaining(remaining_tasks, 1440 - current_start)
    result = [t.get_task_id() for t in ordering]
    return result, get_profit(ordering)

def greedy_critical_time(tasks):
    #deadline / duration
    tasks_sorted = sorted(tasks, key=lambda task: (task.get_deadline()/task.get_duration())) 
    current_start = 0
    ordering = []
    for task in tasks_sorted:
        if (current_start + task.get_duration() <= task.get_deadline()):
            ordering.append(task)
            current_start += task.get_duration()
    remaining_tasks = [t for t in tasks if t not in ordering]
    ordering += fill_remaining(remaining_tasks, 1440 - current_start)
    result = [t.get_task_id() for t in ordering]
    return result, get_profit(ordering), ordering

def greedy_min_slack(tasks):
    #deadline - duration
    tasks_sorted = sorted(tasks, key=lambda task: (task.get_deadline() - task.get_duration())) 
    current_start = 0
    ordering = []
    for task in tasks_sorted:
        if (current_start + task.get_duration() <= task.get_deadline()):
            ordering.append(task)
            current_start += task.get_duration()
    remaining_tasks = [t for t in tasks if t not in ordering]
    ordering += fill_remaining(remaining_tasks, 1440 - current_start)
    result = [t.get_task_id() for t in ordering]
    return result, get_profit(ordering), ordering

def check_validity(results, tasks):
    time = 0
    used = []
    for i in results:
        if i in used: # no repeats
            return False
        time += tasks[i-1].get_duration()
        used.append(i)
    if time > 1440: # don't go past 1440
        return False
    return True


def dp(tasks):
  n = len(tasks)
  arr = [[0 for t in range(1440)] for i in range(n)]
  for t in range(1440):
    arr[0][t] = 0
  deadline_sorted = sorted(tasks, key=lambda task: (task.get_deadline()))
  
  
  def printopt(i, t, lst):
    task = deadline_sorted[i]
    t_i = task.get_duration()
    d_i = task.get_deadline()
    if i == 0:
      return  #, get_profit(lst), [t.get_task_id() for t in lst]
    if arr[i][t] == arr[i-1][t]:
      printopt(i-1, t, lst)
    else:
      tp = min(t, d_i) - t_i
      lst.append(task)
      printopt(i-1, tp, lst)
  
  for i in range(n):
    task = deadline_sorted[i]
    t_i = task.get_duration()
    d_i = task.get_deadline()
    p_i = task.get_max_benefit()
    for t in range(1440):
      tp = min(t, d_i) - t_i
      if tp < 0:
        arr[i][t] = arr[i-1][t]
      else:
        arr[i][t] = max(arr[i-1][t], p_i + arr[i-1][tp])
      
  lst = []
  printopt(n-1, 1439, lst)
  lst = lst[::-1]
  return lst
 

    
if __name__ == '__main__':
    for dirname in ['small', 'medium', 'large']:
        for input_path in os.listdir('inputs/' + dirname):
            if input_path[0] == '.':
                continue
            output_path = 'outputs/' + dirname + '/' + input_path[:-3] + '.out'
            tasks = read_input_file('inputs/' + dirname + '/' + input_path)
            output = solve(tasks)
            write_output_file(output_path, output)
        
            