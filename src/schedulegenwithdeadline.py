""" Defined function for static initial data for Production Planning """

# importing libraries
import pandas as pd
# import openpyxl as xl
import json


def data_from_json():
    # Opening JSON file
    f = open('data.json')

    # returns JSON object as
    # a dictionary
    data = json.load(f)

    # Iterating through the json
    # list
    for i in data['Flow Sequence']:
        print(i)

    # Closing file
    f.close()
    return data


def json_to_df(json_data):
    """ convert json into excel """
    dict_data = {}
    for key in json_data.keys():
        dict_data[key] = pd.DataFrame(json_data.get(key)).T
    return dict_data


''' Solving trafficscheduling problem by genetic algorithm '''

# importing required modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import chart_studio.plotly as py
import plotly.figure_factory as ff
import datetime
import time
import copy


def json_to_df(json_data):
    """ convert json into excel """
    dict_data = {}
    for key in json_data.keys():
        dict_data[key] = pd.DataFrame(json_data.get(key)).T

    return dict_data


class Flow:
    def __init__(self, identifier, sender, receiver, size, sequence, deadline):
        self.identifier = identifier
        self.sender = sender
        self.receiver = receiver
        self.size = size
        self.endToEndDelay = 0
        self.injectTime = 0
        self.arrival = 0
        self.deadline = deadline
        self.sequence = sequence
        self.periodInterval = 1  # 1 Millisecond
        self.nodeType = 0  # 0- Switch, 1-Source, 2-Destination
        self.previous = None
        self.subflows = []

    def get_identifier(self):
        return self.identifier

    def display(self, nodetype):
        self.nodeType = nodetype

    def set_previousflow(self, Flow):
        self.previous = Flow

    def add_SubFlow(self, flow):
        self.subflows.append(flow)

    def get_endToEndDelay(self):
        return self.endToEndDelay


##################FLOWS##################################
#  f1 : ES1->SW1 (Link1), SW1->ES3 (Link2), Period: 1000 ms, Nw Delay 20ms
#  f2 : ES1-> SW1(Link1), SW1->SW2 (Link3),SW2->ES3 (Link4) , Period 2000ms, Nw Delay-10ms
#  f3 : ES2->SW2(link5), SW2->SW1(Link6), SW1->ES3 (Link2) , Period: 1000ms, 10
#  f4 : ES2->SW2 (Link5), SW2->SW1 (Link6), SW1->ES3 (Link2), Period: 2000ms, 20ms


# Flows Defined - Traffic flowing across the network
flow1 = Flow(0, "es1", "es3", 69632, 1, 100)  # Id, sender, receiver, size, sequence, deadline
flow2 = Flow(1, "es1", "es3", 69632, 2, 200)
flow3 = Flow(2, "es2", "es3", 69632, 3, 100)
flow4 = Flow(3, "es2", "es3", 69632, 4, 200)
flow5 = Flow(4, "es2", "es3", 69632, 5, 100)
flow6 = Flow(5, "es2", "es3", 69632, 6, 200)

flows = [flow1, flow2, flow3, flow4, flow5, flow6]


def prepare_initial_population(population_size, population_list, num_gene, num_flows):
    for i in range(population_size):
        nxm_random_num = list(np.random.permutation(num_gene))  # generate a random permutation of 0 to num_job*num_mc-1
        population_list.append(nxm_random_num)  # add to the population_list
        for j in range(num_gene):
            population_list[i][j] = population_list[i][
                                        j] % num_flows  # convert to flow number format, every flow appears m times
    #print("Population List:", population_list)
    return population_list


def two_point_crossover(population_list, population_size, crossover_rate, num_gene):
    parent_list = copy.deepcopy(population_list)
    offspring_list = copy.deepcopy(
        population_list)  # generate a random sequence to select the parent chromosome to crossover
    pop_random_size = list(np.random.permutation(population_size))

    for size in range(int(population_size / 2)):
        crossover_prob = np.random.rand()
        if crossover_rate >= crossover_prob:
            parent_1 = population_list[pop_random_size[2 * size]][:]
            parent_2 = population_list[pop_random_size[2 * size + 1]][:]

            child_1 = parent_1[:]
            child_2 = parent_2[:]
            cutpoint = list(np.random.choice(num_gene, 2, replace=False))
            cutpoint.sort()

            child_1[cutpoint[0]:cutpoint[1]] = parent_2[cutpoint[0]:cutpoint[1]]
            child_2[cutpoint[0]:cutpoint[1]] = parent_1[cutpoint[0]:cutpoint[1]]
            offspring_list[pop_random_size[2 * size]] = child_1[:]
            offspring_list[pop_random_size[2 * size + 1]] = child_2[:]
    return offspring_list, parent_list


def perform_mutations(offspring_list, mutation_rate, num_gene, num_mutation_jobs):
    for off_spring in range(len(offspring_list)):

        """ Mutations """
        mutation_prob = np.random.rand()
        if mutation_rate >= mutation_prob:
            m_change = list(
                np.random.choice(num_gene, num_mutation_jobs, replace=False))  # chooses the position to mutation
            t_value_last = offspring_list[off_spring][
                m_change[0]]  # save the value which is on the first mutation position
            for i in range(num_mutation_jobs - 1):
                offspring_list[off_spring][m_change[i]] = offspring_list[off_spring][
                    m_change[i + 1]]  # displacement
            # move the value of the first mutation position to the last mutation position
            offspring_list[off_spring][m_change[num_mutation_jobs - 1]] = t_value_last
    return offspring_list


def checkfitness_and_calculate_makespan(parent_list, offspring_list, population_size, num_flows, num_links,
                                        process_time, flow_sequence):
    """ fitness value (calculate makespan) """
    total_chromosome = copy.deepcopy(parent_list) + copy.deepcopy(
        offspring_list)  # parent and offspring chromosomes combination
    chrom_fitness, chrom_fit = [], []
    total_fitness = 0
    population_list_fit = []
    for pop_size in range(population_size * 2):
        f_keys = [j for j in range(num_flows)]
        key_count = {key: 0 for key in f_keys}
        f_count = {key: 0 for key in f_keys}
        l_keys = [j + 1 for j in range(num_links)]
        l_count = {key: 0 for key in l_keys}
        """for i in total_chromosome[pop_size]:
            gen_t = int(process_time[i][key_count[i]])
            gen_l = int(flow_sequence[i][key_count[i]])
            # Check for flow and deadline here
            # if this flow then check if the total timeline is less than deadline,
            # if not do not include it in f_count / something like that
            flow_i = flows[i]
            # flow_i.endToEndDelay = f_count[i] + gen_t
            if ((f_count[i] + gen_t) <= flow_i.deadline) and ((l_count[gen_l] + gen_t) <= flow_i.deadline):  # for a valid schedule, the end to end delay for one flow should be within deadline
                #print("Fit case, Flow i", flow_i.identifier, ":", f_count[i] + gen_t, ":", flow_i.deadline)
                population_list_fit.append(1)
                f_count[i] = f_count[i] + gen_t
                l_count[gen_l] = l_count[gen_l] + gen_t
                if l_count[gen_l] < f_count[i]:  # Check if
                    l_count[gen_l] = f_count[i]
                elif (l_count[gen_l] > f_count[i]):
                    f_count[i] = l_count[gen_l]
                # flow_i.endToEndDelay = f_count[i]
                    #print("Special case: Flow i", flow_i.identifier, ":", f_count[i], "Deadline:", flow_i.deadline)

            else:

                population_list_fit.append(0)

            key_count[i] = key_count[i] + 1
            # print("End to end delay:",flow_i.endToEndDelay)
            # else:  # unfit case
            #   continue

            # print("Flow Count", f_count)
        # print("Link count", l_count)
            if (max(f_count.values()) != 0):

                makespan = max(f_count.values())
                chrom_fitness.append(1 / makespan)
                chrom_fit.append(makespan)
                total_fitness = total_fitness + chrom_fitness[pop_size]
            else:
                makespan = 99999999999
                chrom_fitness.append(1/makespan)
                chrom_fit.append(makespan)
                total_fitness = total_fitness + chrom_fitness[pop_size]
            """
        makespan = calculate_fitness_for_chromosome(total_chromosome[pop_size],process_time,key_count,flow_sequence,f_count,l_count,population_list_fit)
        chrom_fitness.append(1 / makespan)
        chrom_fit.append(makespan)
        total_fitness = total_fitness + chrom_fitness[pop_size]
    return total_fitness, chrom_fitness, chrom_fit, total_chromosome, population_list_fit

def calculate_fitness_for_chromosome(current_chromosome,process_time,key_count,flow_sequence,f_count,l_count,population_list_fit):
    """
    Seperate Fitness function per chromosome. Then we can
    check for fitness and include only the fit ones as part of selection
    Fitness here includes minimizing makespan while also meeting deadline for flow
    """
    for i in current_chromosome:
        gen_t = int(process_time[i][key_count[i]])
        gen_l = int(flow_sequence[i][key_count[i]])
        # Check for flow and deadline here
        # if this flow then check if the total timeline is less than deadline,
        # if not do not include it in f_count / something like that
        flow_i = flows[i]
        # flow_i.endToEndDelay = f_count[i] + gen_t
        if ((f_count[i] + gen_t) <= flow_i.deadline) and ((l_count[
                                                               gen_l] + gen_t) <= flow_i.deadline):  # for a valid schedule, the end to end delay for one flow should be within deadline
            # print("Fit case, Flow i", flow_i.identifier, ":", f_count[i] + gen_t, ":", flow_i.deadline)
            population_list_fit.append(1)
            f_count[i] = f_count[i] + gen_t
            l_count[gen_l] = l_count[gen_l] + gen_t
            if l_count[gen_l] < f_count[i]:  # Check if
                l_count[gen_l] = f_count[i]
            elif (l_count[gen_l] > f_count[i]):
                f_count[i] = l_count[gen_l]
            # flow_i.endToEndDelay = f_count[i]
            # print("Special case: Flow i", flow_i.identifier, ":", f_count[i], "Deadline:", flow_i.deadline)

        else:

            population_list_fit.append(0)

        key_count[i] = key_count[i] + 1
        # print("End to end delay:",flow_i.endToEndDelay)
        # else:  # unfit case
        #   continue

        # print("Flow Count", f_count)
        # print("Link count", l_count)
        if (max(f_count.values()) != 0):

            makespan = max(f_count.values())
            #chrom_fitness.append(1 / makespan)
            #chrom_fit.append(makespan)
            #total_fitness = total_fitness + chrom_fitness[pop_size]
        else:
            makespan = 99999999999
            #chrom_fitness.append(1 / makespan)
            #chrom_fit.append(makespan)
            #total_fitness = total_fitness + chrom_fitness[pop_size]
    return makespan

def plot_gantt_chart(num_links, num_flows, sequence_best, process_time, flow_sequence):
    l_keys = [j + 1 for j in range(num_links)]
    f_keys = [j for j in range(num_flows)]
    key_count = {key: 0 for key in f_keys}
    f_count = {key: 0 for key in f_keys}
    l_count = {key: 0 for key in l_keys}
    f_record = {}
    for i in sequence_best:
        gen_t = int(process_time[i][key_count[i]])
        gen_l = int(flow_sequence[i][key_count[i]])
        # if (gen_m != 0):

        #flow_i = flows[i - 1]
        # flow_i.endToEndDelay = f_count[i] + gen_t
        #if (f_count[i] + gen_t) <= flow_i.deadline:  # for a valid schedule, the end to end delay for one flow should be within deadline
        #   print("Fit case, Flow i", flow_i.identifier, ":", f_count[i] + gen_t, ":", flow_i.deadline)
        f_count[i] = f_count[i] + gen_t
        l_count[gen_l] = l_count[gen_l] + gen_t
        if l_count[gen_l] < f_count[i]:  # Check if
            l_count[gen_l] = f_count[i]
        elif (l_count[gen_l] > f_count[i]):
             f_count[i] = l_count[gen_l]

        # if flow_i.endToEndDelay <= flow_i.deadline:
        #    print("Fit case, Flow i", flow_i.identifier, ":", flow_i.endToEndDelay)
        # f_count[i] = f_count[i] + gen_t

        # f_count[i] = f_count[i] + gen_t

        # l_count[gen_l] = l_count[gen_l] + gen_t

        # if l_count[gen_l] < f_count[i]:
        #    l_count[gen_l] = f_count[i]
        # elif l_count[gen_l] > f_count[i]:
        #   f_count[i] = l_count[gen_l]
        # else:
        # j_count[i] =  j_count[i] + gen_t
        # m_count[gen_m] =  0

        start_time = str(datetime.timedelta(
            seconds=f_count[i] - process_time[i][key_count[i]]))  # convert seconds to hours, minutes and seconds
        end_time = str(datetime.timedelta(seconds=f_count[i]))

        f_record[(i, gen_l)] = [start_time, end_time]

        key_count[i] = key_count[i] + 1
    # print("J Record",j_record)
    print ("f_record:", f_record )
    df = []
    for link_key in l_keys:
        for flow_key in f_keys:
            # if ( m!=3 & j!=0):
            # print(j_record[j, m])
            flow=flows[flow_key]
            df.append(dict(Task='Link %s' % (link_key), Start='2020-02-01 %s' % (str(f_record[(flow_key, link_key)][0])), \
                           Finish='2020-02-01 %s' % (str(f_record[(flow_key, link_key)][1])), Resource='Flow %s Deadline %d' % (flow_key + 1,flow.deadline)))

    df_ = pd.DataFrame(df)
    df_.Start = pd.to_datetime(df_['Start'])
    df_.Finish = pd.to_datetime(df_['Finish'])
    start = df_.Start.min()
    end = df_.Finish.max()

    df_.Start = df_.Start.apply(lambda x: x.strftime('%Y-%m-%dT%H:%M:%S'))
    df_.Finish = df_.Finish.apply(lambda x: x.strftime('%Y-%m-%dT%H:%M:%S'))
    print("Df", df)
    data = df_.to_dict(orient='records')

    final_data = {
        'start': start.strftime('%Y-%m-%dT%H:%M:%S'),
        'end': end.strftime('%Y-%m-%dT%H:%M:%S'),
        'data': data}

    fig = ff.create_gantt(df, index_col='Resource', show_colorbar=True, group_tasks=True, showgrid_x=True,
                          title='Traffic Schedule')
    fig.show()
    return final_data, df


def traffic_schedule(data_dict, population_size=30, crossover_rate=0.8, mutation_rate=0.2, mutation_selection_rate=0.2,
                     num_iteration=3000):
    """ initialize genetic algorithm parameters and read data """
    data_json = json_to_df(data_dict)
    flow_sequence_tmp = data_json['Flow Sequence']
    process_time_tmp = data_json['Processing Time']

    df_shape = process_time_tmp.shape

    num_links = df_shape[1]  # number of links
    num_flows = df_shape[0]  # number of flows
    print("Num Links,", num_links, "Num of Flows", num_flows)
    num_gene = num_links * num_flows  # number of genes in a chromosome
    num_mutation_jobs = round(num_gene * mutation_selection_rate)

    process_time = [list(map(int, process_time_tmp.iloc[i])) for i in range(num_flows)]
    flow_sequence = [list(map(int, flow_sequence_tmp.iloc[i])) for i in range(num_flows)]

    # start_time = time.time()

    Tbest = 999999999999999

    best_list, best_obj = [], []
    population_list = []
    makespan_record = []

    # Initial Population
    population_list = prepare_initial_population(population_size, population_list, num_gene, num_flows)
    print("Population List:", population_list)
    # Iterations start here
    for iteration in range(num_iteration):
        Tbest_now = 99999999999

        """ Two Point Cross-Over """
        offspring_list, parent_list = two_point_crossover(population_list, population_size, crossover_rate, num_gene)

        for pop in range(population_size):

            """ Repairment """
            job_count = {}
            larger, less = [], []  # 'larger' record jobs appear in the chromosome more than pop times, and 'less' records less than pop times.
            for job in range(num_flows):
                if job in offspring_list[pop]:
                    count = offspring_list[pop].count(job)
                    pos = offspring_list[pop].index(job)
                    job_count[job] = [count, pos]  # store the above two values to the job_count dictionary
                else:
                    count = 0
                    job_count[job] = [count, 0]

                if count > num_links:
                    larger.append(job)
                elif count < num_links:
                    less.append(job)

            for large in range(len(larger)):
                change_job = larger[large]
                while job_count[change_job][0] > num_links:
                    for les in range(len(less)):
                        if job_count[less[les]][0] < num_links:
                            offspring_list[pop][job_count[change_job][1]] = less[les]
                            job_count[change_job][1] = offspring_list[pop].index(change_job)
                            job_count[change_job][0] = job_count[change_job][0] - 1
                            job_count[less[les]][0] = job_count[less[les]][0] + 1
                        if job_count[change_job][0] == num_links:
                            break

        offspring_list = perform_mutations(offspring_list, mutation_rate, num_gene, num_mutation_jobs)

        total_fitness, chrom_fitness, chrom_fit, total_chromosome,  population_list_fit = checkfitness_and_calculate_makespan(
            parent_list, offspring_list, population_size, num_flows, num_links, process_time, flow_sequence)
        #print ("Chromosome:",total_chromosome, "Makespan:", makespan  )
        """ Selection (roulette wheel approach) """
        #pk, qk = [], []

        #for size in range(population_size * 2):
        #    pk.append(chrom_fitness[size] / total_fitness)
        #for size in range(population_size * 2):
         #   cumulative = 0

          #  for j in range(0, size + 1):
           #     cumulative = cumulative + pk[j]
           # qk.append(cumulative)

        #selection_rand = [np.random.rand() for i in range(population_size)]

        #for pop_size in range(population_size):
         #   if selection_rand[pop_size] <= qk[0]:
          #      population_list[pop_size] = copy.deepcopy(total_chromosome[0])
          #  else:
           #     for j in range(0, population_size * 2 - 1):
            #        if selection_rand[pop_size] > qk[j] and selection_rand[pop_size] <= qk[j + 1]:
             #           population_list[pop_size] = copy.deepcopy(total_chromosome[j + 1])
            #            break
        #equence_now=None
        """ comparison """
        for pop_size in range(population_size * 2):
            if (population_list_fit [pop_size]==1):
                #print("Compare inside -<")
                chom_fit_current = chrom_fit[pop_size]
                if chom_fit_current < Tbest_now:
                    Tbest_now = chrom_fit[pop_size]
                    sequence_now = copy.deepcopy(total_chromosome[pop_size])
            #else:
            #    print("Compare outside ->")
        if (Tbest_now <= Tbest and sequence_now != None ):
            Tbest = Tbest_now
            sequence_best = copy.deepcopy(sequence_now)

        makespan_record.append(Tbest)

    """ Results - Makespan """

    print("optimal sequence", sequence_best)
    print("optimal value:%f" % Tbest)
    print("\n")
    # print('the elapsed time:%s'% (time.time() - start_time))

    # %matplotlib inline
    plt.plot([i for i in range(len(makespan_record))], makespan_record, 'b')
    plt.ylabel('makespan', fontsize=15)
    plt.xlabel('generation', fontsize=15)
    plt.show()

    print(makespan_record)

    """ plot gantt chart """
    final_data, df = plot_gantt_chart(num_links, num_flows, sequence_best, process_time, flow_sequence)

    # iplot(fig, filename='GA_job_shop_scheduling')
    print(final_data)
    return final_data, df


""" Job_Shop_Schedule """

# data = data_excel_json('data/JSP_dataset.xlsx')
data = data_from_json()
print("Data", data)
schedule = traffic_schedule(data_dict=data)
print("Schedule", schedule[0])

# import chart_studio.plotly as py
# import plotly.figure_factory as ff

# df = schedule[1]
# fig = ff.create_gantt(df, index_col='Resource', show_colorbar=True, group_tasks=True, showgrid_x=True, title='Job shop Schedule')
# fig.show()
