from parse import read_input_file, write_output_file
import os

def solve(tasks):
    """
    Args:
        tasks: list[Task], list of igloos to polish
    Returns:
        output: list of igloos in order of polishing  
    """        
    deadline = [] #t = deadline array; [deadline, task]
    duration = [] #d = duration array [duration, task]
    profit = [] #p = profit array 
    n = len(tasks)
    
    #sort by profit
    tasks = sorted(tasks, key=lambda task: task.get_max_benefit())[::-1]
    for i in range(n):
      duration.append(tasks[i].get_duration())
      deadline.append(tasks[i].get_deadline())
      profit.append(tasks[i].get_max_benefit())

    result = []

    #for each item sorted on profit, add task ending at it's deadline (or move earlier if needed)
    #for each item sorted on profit, add task to front
    current_start = 0
    for i in range(len(tasks)):
        if (current_start + duration[i] <= deadline[i]):
            result.append(tasks[i].get_task_id())
            current_start = current_start + duration[i]
    
    return result

    pass




# Here's an example of how to run your solver.
# if __name__ == '__main__':
#     for input_path in os.listdir('inputs/'):
#         output_path = 'outputs/' + input_path[:-3] + '.out'
#         tasks = read_input_file(input_path)
#         output = solve(tasks)
#         write_output_file(output_path, output)

if __name__ == '__main__':
    output_path = 'test.out'
    tasks = read_input_file('inputs/large/large-1.in')
    output = solve(tasks)
    write_output_file(output_path, output)