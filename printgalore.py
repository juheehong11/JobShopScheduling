from parse import read_input_file, write_output_file
import os
from ortools.algorithms import pywrapknapsack_solver # https://developers.google.com/optimization/bin/knapsack
# import numpy as np

def solve(tasks, special=False): #probably should get rid of this special thing
    """
    Args:
        tasks: list[Task], list of igloos to polish 
    Returns:
        output: list of igloos in order of polishing  
    """
    best_result = None
    best_profit = 0
    n = len(tasks)
    used_greedy = False
    duration = []
    deadline = []
    profit = []

    skipknap = False
    #analyze for duplicates/ same properties
    for i in range(n):
      duration.append(tasks[i].get_duration())
      deadline.append(tasks[i].get_deadline())
      profit.append(tasks[i].get_max_benefit())
        

    print(sum(profit)/len(profit))
    if int(sum(profit)/len(profit)) == int(profit[1]) == int(profit[50]):
        skipknap = True
    #all same duration:
    # if avg(duration) == duration[1]:
    #     skipknap = True
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
            print('CHANGED SKIPKNAP BACK TO FALSE')
            break
        #print('d :' + str(deaddurprof[key]))
        if deaddurprof[key] > 6:
            skipknap = True
            break

    #print(len(durprof))
    if len(durprof) < 8:
        print('TOO FEW KEYS----------------------')
        skipknap = True
    
    if skipknap:
        print("skipknap------------------------------")
    
    # if not special:
    if not skipknap:
        for size in range(n//8,n//2+1,n//8):
            id_ordering, profit, tasks_ordering = knapsack_solver(tasks,size)
            assert profit == get_profit(tasks_ordering)
            if profit > best_profit:
                best_profit = profit
                best_result = id_ordering
                best_ordering = tasks_ordering
    count = 0
    for id_ordering, profit, tasks_ordering in [greedy_profit_duration(tasks), greedy_duration_deadline(tasks), greedy_duration_deadline(tasks), greedy_critical_time(tasks),
                                                greedy_min_slack(tasks), greedy_decay(tasks), greedy_decay_profit_duration(tasks)]: #backtrack_profit_duration(tasks)]:
        assert profit == get_profit(tasks_ordering)
        if profit > best_profit:
            #used_greedy = True
            #print(profit)
            #print(id_ordering)
            best_profit = profit
            best_result = id_ordering
            best_ordering = tasks_ordering
            print('USED GREEDY: ' + str(count))
            if count > 5:
                print('USED MADDY --------------------------')
            if count > 6:
                print('USED JESSIE -------------------------')
        count += 1
    
    dp_ordering = dp(tasks)
    dp_profit = get_profit(dp_ordering)
    if dp_profit > best_profit:
        best_profit = dp_profit
        best_ordering = dp_ordering
        best_result = [t.get_task_id() for t in dp_ordering]
        print('USED DP---------------------------')

    #if not used_greedy:
        #print('USED KNAPSACK')
    print('profit: ' + str(best_profit))
    #print(best_result)
    #print(best_profit)
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

    #elementary counter here
    #start clock

    for i in range(0, len(tasks), knapsack_size):
        tasks_group = tasks_sorted_deadline[i:min(i+knapsack_size,len(tasks_sorted_deadline))]
        values = [t.get_max_benefit() for t in tasks_group]
        weights = [[t.get_duration() for t in tasks_group]]
        capacities = [min(tasks_group[-1].get_deadline(), total_time) - current_start]
        solver.Init(values, weights, capacities)

        #https://stackoverflow.com/questions/366682/how-to-limit-execution-time-of-a-function-call
        # try:
        #     with time_limit(10):
        #         computed_value = solver.Solve()
        # except TimeoutException as e:
            # return [], 0

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
    #print(sum([t.get_duration() for t in ordering]))
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
        #if testing:
            #print(current_time)
            #print(str(task.get_task_id()) + ': ' + str(total_profit))
    return total_profit
    
def fill_remaining(tasks_remaining, time_remaining, from_decay=False):
    # compares two different methods of filling in remaining time:
    # tries fill in backwards from 1440, latest deadline first, and tries greedy_decay
    tasks_sorted = sorted(tasks_remaining, key=lambda task: (-task.get_deadline(), -task.get_max_benefit()))
    ordering = []
    temp_time = time_remaining
    #orig_remain = True
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
                #orig_remain = False
                ordering = other_task_ordering
    return ordering

def greedy_decay(tasks, time_remaining=1440):
    # different from other greedys because it allows you to go past deadline
    profit_duration = sorted(tasks, key=lambda task: (-task.get_max_benefit()/task.get_duration(), task.get_deadline()))
    profit_forwards = sorted(tasks, key=lambda task: (-task.get_max_benefit(), task.get_deadline()))
    sorted_deadline = sorted(tasks, key=lambda task: (task.get_deadline(), -task.get_max_benefit()))
    sorted_duration = sorted(tasks, key=lambda task: (task.get_duration(), task.get_deadline()))
    #count = 0
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
                                     
def backtrack_profit_duration(tasks):
    # when to backtrack? 
    # when we find another task that has greater profit/duration that we can't take due to time constraints
    # eg. if current_start + task.get_duration() > task.get_deadline(), remove the previous task from 
    # both our list and choice of options 

    counter_backtrack = 0
    
    tasks_sorted = sorted(tasks, key=lambda task: task.get_max_benefit() / task.get_duration())[::-1]

    previous_task = tasks_sorted[0]
    current_start = tasks_sorted[0].get_duration()
    ordering = [tasks_sorted[0]]
    
    for i in range(1, len(tasks_sorted)):
        task = tasks_sorted[i]
        unit_prof = task.get_max_benefit() / task.get_duration()
        if current_start + task.get_duration() > task.get_deadline() and unit_prof >= previous_task.get_max_benefit()/previous_task.get_duration():
            while ordering and current_start + task.get_duration() > task.get_deadline():
                temp = ordering.pop()
                current_start -= temp.get_duration()
            counter_backtrack += 1
        #else if current_start + task.get_duration() <= task.get_deadline():
        ordering.append(task)
        current_start += task.get_duration()
        previous_task = task
    remaining_tasks = [t for t in tasks if t not in ordering]
    ordering += fill_remaining(remaining_tasks, 1440 - current_start)
    result = [t.get_task_id() for t in ordering]
    return result, get_profit(ordering), ordering                                

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
            print('repeat')
            return False
        time += tasks[i-1].get_duration()
        used.append(i)
        #print(i)
        #print(time)
    if time > 1440: # don't go past 1440
        print('overtime')
        return False
    return True


def dp(tasks):
  n = len(tasks)
  arr = [[0 for t in range(1440)] for i in range(n)]
  #print(arr)
  #arr(i,t) = a feasible schedule in which only jobs from [1, i] are scheduled and all jobs finish by time t.
  for t in range(1440):
    arr[0][t] = 0
  deadline_sorted = sorted(tasks, key=lambda task: (task.get_deadline()))
  
  #tp = min(t, d_i) - t_i
  #if tp < 0 then arr[i][t] = arr[i-1][t]
  #if tp >= 0 then arr[i][t] = max(arr[i-1][t], p_i + arr[i-1][tp])
  
  def printopt(i, t, lst):
    task = deadline_sorted[i]
    t_i = task.get_duration()
    d_i = task.get_deadline()
    if i == 0:
      #print([t.get_task_id() for t in lst])
      #print(get_profit(lst))
      #print(lst)
      #print(lst == None)
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
    #task1 = tasks[i+1]
    #t1 = task1.get_duration()
    #d1 = task1.get_deadline()
    #p1 = task1.get_max_benefit()
    for t in range(1440):
      tp = min(t, d_i) - t_i
      if tp < 0:
        arr[i][t] = arr[i-1][t]
      else:
        #print(i, t, i-1, tp)
        arr[i][t] = max(arr[i-1][t], p_i + arr[i-1][tp])
      #if(arr[i][t] != -1 and t + t1 <= d1):
        #arr[i+1][t+t1] = max(arr[i+1][t+t1], arr[i][t] + p1)

  lst = []
  printopt(n-1, 1439, lst)
  lst = lst[::-1]
  #print("dp profit", get_profit(lst))
  return lst
 

    
if __name__ == '__main__':
    for dirname in ['small', 'medium', 'large']:
        #count = 30
        for input_path in os.listdir('inputs/' + dirname):
            # input_path = 'small-161.in'
            print(input_path)
            if input_path[0] == '.':
                continue
            output_path = 'outputs/' + dirname + '/' + input_path[:-3] + '.out'
            tasks = read_input_file('inputs/' + dirname + '/' + input_path)
            # just doing this for now because can't figure out why these three don't work for knapsack
            #if input_path == 'medium-54.in' or input_path == 'large-2.in' or input_path == 'large-111.in' or input_path == 'medium-111.in' or input_path == 'small-111.in' or input_path == 'small-74.in' or input_path == 'small-235.in' or input_path == 'medium-93.in' or input_path == 'medium-77.in':
                #output = solve(tasks, True)
            #else:
            output = solve(tasks)
            write_output_file(output_path, output)
            # break
            #count -= 1
            #if count == 0:

        
            

# Here's an example of how to run your solver.
#if __name__ == '__main__':
    #for input_path in os.listdir('inputs'):
        #output_path = 'outputs/' + input_path[:-3] + '.out'
        #tasks = read_input_file(input_path)
        #output = solve(tasks)
        #write_output_file(output_path, output)

# if __name__ == '__main__':
#     output_path = 'test.out'
#     tasks = read_input_file('test.in')
#     output = solve(tasks)
