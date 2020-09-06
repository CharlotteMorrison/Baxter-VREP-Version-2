from reports import Reports
from graphs import Graphs
import random
import globals
import constants as cons


if __name__ == "__main__":
    # create report
    # report = Reports()
    report = cons.TD3_REPORT
    graph = Graphs()

    # fills reports with dummy values for testing
    counter = 0
    anneal = .001
    for i in range(10000):
        for j in range(random.randint(1, 25)):
            counter += 1
            anneal += (.00001 * random.randint(-1, 2))
            report.write_report_step(i,
                                     counter,
                                     random.random() + anneal,
                                     random.random() + anneal,
                                     random.random() + anneal,
                                     False,
                                     'date')
            report.write_report_actor(i, counter, random.random(), random.random())
            report.write_report_critic(i, counter, random.random(), random.random())
            report.write_report_error(i, counter, random.random())

    graph.update_step_list_graphs()







