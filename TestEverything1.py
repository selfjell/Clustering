import KMeans4 as km
import test as gm

# Global lists
scaler_list = ['minmax', 'robust', 'standard', 'norm']
ordered_list = [True, False]
# If test with comp 2-7 (files_names_2range_7comp), then range 30 is as good as 100 based on previous test
seed_list = range(50)

# [True, False] = 7and2comp
# [True] = 2comp_only
pca_list = [True]

# for testing with all comps
pca_list2 = [2]

km_results = []
gm_results = []
gm_covar_list = ['full', 'tied', 'diag', 'spherical']
file_names_7and2comp = ["SUPER_TEST_KMEANS.txt", "SUPER_TEST_GM.txt"]
file_names_2comp = ["SUPER_TEST_KMEANS_Only2comp.txt", "SUPER_TEST_GM_Only2comp.txt"]
file_names_2range7comp = ["SUPER_TEST_KMEANS_ALLcomp.txt", "SUPER_TEST_GM_ALLcomp.txt"]


# writes scores to a txt file.
def write_scores(results, km_or_gm):

    # argument km_or_gm decides which file to write to
    if km_or_gm == 'km':
        file = open(file_names_2comp[0], "w+")
    else:
        file = open(file_names_2comp[1], "w+")

    file.write(km_or_gm + ' - SUPER_TEST')
    file.write('\n')

    # Goes through the results and writes them to the file
    for scaler_results in results:
        title = scaler_results[0]
        info = scaler_results[1]
        file.write(title + '\n')
        file.write('best: ' + str(info.pop(0)))
        file.write('---------' + '\n')
        for line in info:
            file.write(line)
        file.write('\n')


# for each scaler, check all scores with varying: seed, pca/no-pca, ordered/unordered options
def test_scaler(scaler):

    # Contains the (seed, pca_string, ordered_string, score) in string format
    km_info_list = []

    # Highest score for km with the given scaler
    km_best_score = 0

    # The info (seed, pca_string, ordered_string, score) for the instance with highest score
    km_best_info = ''

    # Same as the aboves
    gm_info_list = []
    gm_best_score = 0
    gm_best_info = ''

    # Main loop for "brute-forcing" to a good score with the scaler
    for seed in seed_list:
        for pca in pca_list2:
            for ordered in ordered_list:

                # The score (given arguments) from a method in KMeans3.py
                score = km.run_everything3(scaler, seed, pca, ordered)

                gm_scores = []
                for gm_type in gm_covar_list:
                    # The score (given arguments and covar_list) from a method in test.py
                    score2 = gm.run3(scaler, seed, pca, ordered, gm_type)
                    gm_scores.append(score2)



                # Translating booleans to strings for easier reading in the file
                pca_string = str(pca) + '_components'
                #if pca:
                #    pca_string = '2_components'
                #else:
                #    pca_string = '7_components'

                if ordered:
                    ordered_string = 'scale_first'
                else:
                    ordered_string = 'scale_after'


                # Makes an entry in km_info_list containing all the arguments that were used to obtain the score
                km_info_list.append('%s, %s, %s, %s %s' % (seed, pca_string, ordered_string, score, '\n'))

                # Keeps track of the best km_score
                if score > km_best_score:
                    km_best_score = score
                    km_best_info = '%s, %s, %s, %s %s' % (seed, pca_string, ordered_string, score, '\n')

                # Same as the aboves with gm_info_list and gm_scores
                for score3, covar in zip(gm_scores, gm_covar_list):
                    gm_info_list.append('%s, %s, %s, %s %s %s' % (seed, pca_string, ordered_string, score3, covar, '\n'))
                    if score3 > gm_best_score:
                        gm_best_score = score3
                        gm_best_info = '%s, %s, %s, %s %s %s' % (seed, pca_string, ordered_string, score3, covar, '\n')

    # Adds the best score for the scaler at the beginning of the info_lists
    km_info_list.insert(0, km_best_info)
    gm_info_list.insert(0, gm_best_info)
    return km_info_list, gm_info_list


# Goes through scaler_list and gets results from test_scaler
# Then writes those results to a file
def run_things():
    print("Performing complicated algorithms...")
    print("This could take a few minutes...")
    for scaler in scaler_list:

        # Gets the results into the two lists (from global, see the top of file)
        test_results_km, test_results_gm = test_scaler(scaler)

        km_results.append([scaler, test_results_km])
        gm_results.append([scaler, test_results_gm])

    write_scores(km_results, 'km')
    write_scores(gm_results, 'gm')
    print("Success!")


# Everything is run
run_things()


