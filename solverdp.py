from parse import read_input_file, write_output_file
import os

def solve(tasks):
    """
    Args:
        tasks: list[Task], list of igloos to polish
    Returns:
        output: list of igloos in order of polishing  
    """        
    deadline = [-1] #t = deadline array; [deadline, task]
    duration = [-1] #d = duration array [duration, task]
    profit = [-1] #p = profit array 
    n = len(tasks)
    
    #sort by deadline
    tasks = sorted(tasks, key=lambda task: task.get_deadline())
    for i in range(0, n):
      duration.append(tasks[i].get_duration())
      deadline.append(tasks[i].get_deadline())
      profit.append(tasks[i].get_max_benefit())
      
    
    #dp: row = # of jobs, col = deadline
    dp = [[-1 for i in range(1440)] for _ in range(n)]
    # dp[0][0] = 0.0
    dp[0] = [0.0] * 1440
    B = []
    order = {(i,j):[] for _ in range(1440)} # (i,j) --> [order]
    
    
    # i jobs means up to i jobs 
    for i in range(n):
      for j in range(1400):
        if dp[i][j] != -1 and j + duration[i+1] <= deadline[i+1]:
          dp[i+1][j+duration[i+1]] = max(dp[i+1][j + duration[i+1]], dp[i][j] + profit[i+1])



    # for i in range(n):
        #   for j in range(1400):
        #     #if task i+1 satisfies deadline condition (is doable), 
        #     #try to update condition for what the highest total profit will be for dp[i+1][j+time[i+1]]
        #     if dp[i][j] != -1 and j + duration[i+1] <= deadline[i+1]:
        #       dp[i+1][j+duration[i+1]] = max(dp[i+1][j + duration[i+1]], dp[i][j] + profit[i+1])
        #       # B[(i+1, j+duration[i+1])] = 

    # Here's an example of how to run your solver.
    # if __name__ == '__main__':
    #     for input_path in os.listdir('inputs/'):
    #         output_path = 'outputs/' + input_path[:-3] + '.out'
    #         tasks = read_input_file(input_path)
    #         output = solve(tasks)
    #         write_output_file(output_path, output)