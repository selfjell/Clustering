import KMeans3 as km
import test as gm


scaler_list = ['minmax', 'robust', 'standard', 'norm']
ordered_list = [True, False]
seed_list = range(100)
pca_list = [True, False]
results = []

def write_scores(results):
    file = open("SUPER_TEST_KMEANS.txt", "w+")
    file.write('SUPER_TEST')
    file.write('\n')
    for scaler_results in results:
        title = scaler_results[0]
        info = scaler_results[1]

        file.write(title + '\n')
        file.write('best: ' + str(info.pop(0)))
        file.write('---------' + '\n')
        for line in info:
            file.write(line)





# What do we want to do?
# 1: Check all minmax, ordered, pca
# 2: Check all minmax, ordered, no-pca
# 3: Check all minmax, unordered, pca
# 4: Check all minmax, unordered, no-pca
# 5: Repeat for every type...


def test_scaler(scaler):
    info_list = []
    best_score = 0
    for seed in seed_list:
        for pca in True, False:
            for ordered in True, False:
                score = km.run_everything2(scaler, seed, pca, ordered)

                if pca:
                    pca_string = '2_components'
                else:
                    pca_string = '7_components'
                if ordered:
                    ordered_string = 'scale_first'
                else:
                    ordered_string = 'scale_after'

                info_list.append('%s, %s, %s, %s %s' % (seed, pca_string, ordered_string, score, '\n'))
                if score > best_score:
                    best_score = score
                    best_info = '%s, %s, %s, %s %s' % (seed, pca_string, ordered_string, score, '\n')


    info_list.append('\n')
    info_list.insert(0, best_info)
    return info_list



def run_things():
    for scaler in scaler_list:
        results.append([scaler, test_scaler(scaler)])
    write_scores(results)


run_things()


