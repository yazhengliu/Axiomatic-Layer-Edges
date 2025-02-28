def select_path_number(dataset,targetpath):
    if dataset=='cora':
        return [1,2,3,4,5]

    if dataset=='Chi' or dataset=='NYC':
        return [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

    elif  dataset=='pheme':
        return [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]  # [15, 16, 17, 18, 19]

    elif dataset=='weibo':
        return [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]  # [15, 16, 17, 18, 19]
    elif  dataset=='bitcoinotc':
        if targetpath > 4000:
            return [100, 120, 140, 160, 180]  # [120, 160, 200, 240,280]
        elif 2000 < targetpath <= 4000:
            return [48, 56, 64, 73, 80]  # [40, 60, 80, 100, 120]
        elif 400 < targetpath < 2000:
            return [24, 28, 32, 36, 40]
        else:
            return [4, 8, 12, 16, 20]
        # return [4, 8, 12, 16, 20]

    elif dataset=='bitcoinalpha':
        if targetpath > 4000:
            return [100, 120, 140, 160, 180]  # [120, 160, 200, 240,280]
        elif 2000 < targetpath <= 4000:
            return [48, 56, 64, 73, 80]  # [40, 60, 80, 100, 120]
        elif 400 < targetpath < 2000:
            return [24, 28, 32, 36, 40]
        else:
            return [4, 8, 12, 16, 20]

    elif dataset=='UCI':
            return [4,8,12,16,20]


    elif dataset=='mutag':
        return [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]


    elif dataset=='clintox':
        return [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]

    elif dataset == 'IMDB-BINARY':
        return [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

    elif dataset == 'REDDIT-BINARY':
        return [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    elif dataset=='physics':
        return [4,8,12,16,20]
    elif dataset=='cs':
        return [4, 8, 12, 16, 20]
    elif dataset=='pubmed':
        return [4, 8, 12, 16, 20]





def select_edge_number(dataset,targetpath):
    if dataset=='Chi' or dataset=='NYC':
        return [1,2,3,4,5]

    elif dataset=='pheme':
        return [1,2,3,4,5]
    elif dataset=='weibo':
        return [1,2,3,4,5]

    elif dataset=='mutag' or dataset=='clintox' or dataset=='IMDB-BINARY' or dataset=='REDDIT-BINARY':
        return [1,2,3,4,5]


    elif  dataset=='bitcoinotc':
            if targetpath > 1000:
                return [25, 30, 35, 40, 45]  # [30, 40, 50, 60, 70]
            elif 500 < targetpath <= 1000:
                return [12, 14, 16, 18, 20]  # [10, 15, 20, 25, 30]
            elif 100 < targetpath < 500:
                return [6, 7, 8, 9, 10]
            else:
                return [1, 2, 3, 4, 5]

    elif dataset=='bitcoinalpha':

        if targetpath > 1000:
            return [25, 30, 35, 40, 45]  # [30, 40, 50, 60, 70]
        elif 500 < targetpath <= 1000:
            return [12, 14, 16, 18, 20]  # [10, 15, 20, 25, 30]
        elif 100 < targetpath < 500:
            return [6, 7, 8, 9, 10]
        else:
            return [1, 2, 3, 4, 5]

    elif dataset == 'UCI':
        return [1,2,3,4,5]

