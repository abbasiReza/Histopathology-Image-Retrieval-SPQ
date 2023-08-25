from utils import *
from tqdm import tqdm
from dataset import SingleData
import math

def get_data_list(data_path, ratio=0.95):
    img_list = []
    for root, dirs, files in os.walk(data_path):
        if files == []:
            class_name = dirs
        elif dirs == []:
            for f in files:
                img_path = os.path.join(root, f)
                img_list.append(img_path)

    np.random.seed(1)
    train_img_list = np.random.choice(img_list, size=int(len(img_list)*(1-ratio)), replace=False)
    #print(img_list, train_img_list)
    eval_img_list = list(set(img_list) - set(train_img_list))
    ########add
    half=math.floor(len(eval_img_list)/2)
    print(half)
    eval_=eval_img_list[:half]
    test_=eval_img_list[half:]
    #######
    #return class_name, train_img_list, eval_img_list 
    return class_name, train_img_list

def pqDist_one(C, N_books, g_x, q_x,query_pname,gallery_pname):
    l1, l2 = C.shape
    L_word = int(l2/N_books)
    D_C = T.zeros((l1, N_books), dtype=T.float32)

    q_x_split = T.split(q_x, L_word, 0)
    g_x_split = np.split(g_x.cpu().data.numpy(), N_books, 1)
    C_split = T.split(C, L_word, 1)
    D_C_split = T.split(D_C, 1, 1)

    for j in range(N_books):
        for k in range(l1):
            # D_C_split[j][k] =T.norm(q_x_split[j]-C_split[j][k], 2)
            D_C_split[j][k] = T.norm(q_x_split[j]-C_split[j][k], 2).detach() #for PyTorch version over 1.9
        if j == 0:
            dist = D_C_split[j][g_x_split[j]]
        else:
            dist = T.add(dist, D_C_split[j][g_x_split[j]])
    Dpq = T.squeeze(dist)
    # #####
    # for i in range(len(gallery_pname)):
    #     if query_pname==gallery_pname[i]:
    #         Dpq[i]=float('inf')
    # #####
    return Dpq

def Indexing(C, N_books, X):
    l1, l2 = C.shape
    L_word = int(l2/N_books)
    x = T.split(X, L_word, 1)
    y = T.split(C, L_word, 1)
    for i in range(N_books):
        diff = squared_distances(x[i], y[i])
        arg = T.argmin(diff, dim=1)
        min_idx = T.reshape(arg, [-1, 1])
        if i == 0:
            quant_idx = min_idx
        else:
            quant_idx = T.cat((quant_idx, min_idx), dim=1)
    return quant_idx

def Evaluate_mAP(C, N_books, gallery_codes, query_codes, gallery_labels, query_labels, device,gallery_pname,query_pname, TOP_K=None):
    num_query = query_labels.shape[0]
    mean_AP = 0.0

    with tqdm(total=num_query, desc="Evaluate mAP", bar_format='{desc:<15}{percentage:3.0f}%|{bar:10}{r_bar}') as pbar:
        for i in range(num_query):
            # Retrieve images from database
            retrieval = (query_labels[i, :] @ gallery_labels.t() > 0).float()

            # Arrange position according to hamming distance
            retrieval = retrieval[T.argsort(pqDist_one(C, N_books, gallery_codes, query_codes[i],query_pname[i],gallery_pname))][:TOP_K]

            # Retrieval count
            retrieval_cnt = retrieval.sum().int().item()

            # Can not retrieve images
            if retrieval_cnt == 0:
                continue

            # Generate score for every position
            score = T.linspace(1, retrieval_cnt, retrieval_cnt).to(device)

            # Acquire index
            index = (T.nonzero(retrieval == 1, as_tuple=False).squeeze() + 1.0).float().to(device)

            mean_AP += (score / index).mean()
            pbar.update(1)

        mean_AP = mean_AP / num_query
    return mean_AP

def DoRetrieval(device, args, net, C):
    print("Do Retrieval!")


    class_name, train_img_list = get_data_list('/content/drive/MyDrive/dataSets/CRC/train/',0.0)
    train_data_set=SingleData(class_name, train_img_list, transform=transforms.ToTensor())
    Gallery_loader = T.utils.data.DataLoader(train_data_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # trainset = torchvision.datasets.CIFAR10(root=args.data_dir, train=True, download=args.if_download, transform=transforms.ToTensor())
    # Gallery_loader = T.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    class_test_name, test_img_list = get_data_list('/content/drive/MyDrive/dataSets/CRC/same test/',0.0)
    test_data_set = SingleData(class_test_name, test_img_list, transform=transforms.ToTensor())    
    
    # testset = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, download=args.if_download, transform=transforms.ToTensor())
    Query_loader = T.utils.data.DataLoader(test_data_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    net.eval()
    with T.no_grad():
        with tqdm(total=len(Gallery_loader), desc="Build Gallery", bar_format='{desc:<15}{percentage:3.0f}%|{bar:10}{r_bar}') as pbar:
            for i, data in enumerate(Gallery_loader, 0):
                gallery_x_batch, gallery_y_batch = data[0].to(device), data[1].to(device)
                gallery_y_batch=gallery_y_batch.long()
                outputs = net(gallery_x_batch)
                gallery_c_batch = Indexing(C, args.N_books, outputs[0])
                gallery_y_batch = T.eye(args.num_cls)[gallery_y_batch]
                temp=[]
                for s in data[2]:
                    index1=s.rfind('/')
                    temp.append(s[index1+4:index1+7])
                    
                if i == 0:
                    gallery_c = gallery_c_batch
                    gallery_y = gallery_y_batch
                    gallery_pname=temp
                else:
                    gallery_c = T.cat([gallery_c, gallery_c_batch], 0)
                    gallery_y = T.cat([gallery_y, gallery_y_batch], 0)
                    gallery_pname=gallery_pname + temp
                pbar.update(1)

        with tqdm(total=len(Query_loader), desc="Compute Query", bar_format='{desc:<15}{percentage:3.0f}%|{bar:10}{r_bar}') as pbar:
            for i, data in enumerate(Query_loader, 0):
                query_x_batch, query_y_batch = data[0].to(device), data[1].to(device)
                query_y_batch=query_y_batch.long()
                outputs = net(query_x_batch)
                query_y_batch = T.eye(args.num_cls)[query_y_batch]
                
                temp=[]
                for s in data[2]:
                    index1=s.rfind('/')
                    temp.append(s[index1+4:index1+7])
                    

                if i == 0:
                    query_c = outputs[0]
                    query_y = query_y_batch
                    query_pname=temp
                else:
                    query_c = T.cat([query_c, outputs[0]], 0)
                    query_y = T.cat([query_y, query_y_batch], 0)
                    query_pname=query_pname+temp
                pbar.update(1)

    mAP = Evaluate_mAP(C, args.N_books, gallery_c.type(T.int), query_c, gallery_y, query_y, device,gallery_pname,query_pname, args.Top_N)
    return mAP
