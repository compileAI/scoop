"""clustering_core.py"""

import pandas as pd
import numpy as np
import time
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from scipy.spatial import distance
from scipy.sparse import vstack
from warnings import simplefilter
from scipy.sparse import csr_matrix, vstack

# ignore scikit-learn's token_pattern notice
simplefilter(action="ignore", category=FutureWarning)

# ------------------------------------------------------------------------------
# utilities
# ------------------------------------------------------------------------------

# helper: spherical k-means via unit-norm + vanilla KMeans
def spherical_kmeans(X, n_clusters, random_state=0):
    """
    Run spherical k-means by l2-normalising the data, fitting KMeans,
    and re-normalising centroids.  Returns (labels, centres).
    """
    Xn = normalize(X)  # unit-length samples
    km = KMeans(n_clusters=n_clusters, n_init="auto", random_state=random_state)
    labels = km.fit_predict(Xn)
    centres = normalize(km.cluster_centers_)
    return labels, centres

def read_dataset(file_name, story_label, verbose):
    article_df = pd.read_json(file_name)
    article_df["sentence_embds"] = [np.array(x) for x in article_df["sentence_embds"]]

    tfidf_vectorizer = TfidfVectorizer(
        ngram_range=(1, 2), tokenizer=lambda x: x, lowercase=False, norm=None
    )
    tfidf_vectorizer.fit_transform([sum(k, []) for k in article_df["sentence_tokens"]])
    all_vocab = tfidf_vectorizer.get_feature_names_out().tolist()

    count_vectorizer = CountVectorizer(
        tokenizer=lambda x: x,
        ngram_range=(1, 2),
        vocabulary=list(all_vocab),
        lowercase=False,
    )
    article_df["sentence_TFs"] = [
        count_vectorizer.transform(y) for y in article_df["sentence_tokens"].values
    ]
    article_df["article_TF"] = [sum(a) for a in article_df["sentence_TFs"].values]

    if verbose:
        print(f"{file_name} loaded")
        print(f"articles:{len(article_df)}")
        if story_label:
            print(f"#stories:{len(article_df.story.unique())}")

    return article_df, all_vocab

def get_article_embedding(slide, window, article_df_slides, time_aware, theme_aware, keyword_score, N):
    start_time = time.time()
    if len(slide) < 1:
        return slide, time.time() - start_time      
     
    if theme_aware:
        num_articles = len(window) if len(window) > 0 else len(slide)
        
        if time_aware: #exponential decaying document frequency
            article_df_window = 0 
            for t in range(len(article_df_slides)):
                article_df_window += np.exp(-(len(article_df_slides)-t-1)/len(article_df_slides))*article_df_slides[t]
        else: #document frequency
            article_df_window = np.sum(article_df_slides,axis=0) 
        
        article_idf_window = np.log((num_articles+1)/(article_df_window+1))+1 #inverse document frequency - scikit-learn formual = log((N+1)/(df+1))+1
        article_tf_window = vstack(slide['article_TF'].values) #term frequency
              
        if keyword_score == 'tfidf':
            article_keyword_score_all = article_tf_window.multiply(article_idf_window).tocsr()
        elif keyword_score == 'bm25':
            k1 = 1.2
            b = 0.75
            d = 1.0
            
            avgDL =  np.sum(vstack(window['article_TF'].values))/num_articles if len(window) > 0 else np.sum(vstack(slide['article_TF'].values))/num_articles #average document length
            article_ntf_window = article_tf_window.multiply(1/np.array(1-b+b*np.sum(article_tf_window,axis=1)/avgDL)) # normalized term frequency - pivoted length normalization - eq3 in Yuanhua 2011
            article_ntf_window.data = article_ntf_window.data # shifting - eq4 in Yuanhua 2011
            article_ntf_window.data = ((k1 + 1) * article_ntf_window.data)  / (k1 + article_ntf_window.data)  + d # tf normalization - eq4 in Yuanhua 2011
            article_keyword_score_all = article_ntf_window.multiply(article_idf_window).tocsr()
        
    weighted_embs = []
    num_processed_articles = 0
    for (idx,article) in slide.iterrows():
        if theme_aware:
            article_topN_indices = article_keyword_score_all[num_processed_articles].indices[article_keyword_score_all[num_processed_articles].data.argsort()[:-(N+1):-1]]
            article_topN_scores = article_keyword_score_all[num_processed_articles][:,article_topN_indices]
            sentence_raw_weights = np.array(np.sum(article.sentence_TFs[:,article_topN_indices].multiply(article_topN_scores), axis=1)).ravel() + 1e-5
            sentence_weights = sentence_raw_weights / np.sum(sentence_raw_weights, axis=0)
            num_processed_articles += 1
        else:
            num_sentences = len(article['sentences'])
            sentence_weights = [1/num_sentences]* num_sentences 
        weighted_embs.append(np.matmul(sentence_weights,article.sentence_embds))
    
    slide['embedding'] = weighted_embs
    
    return slide, time.time() - start_time

def get_cluster_theme(window, window_size, to_date,
                      time_aware, cluster_tf_sum_dics,
                      keyword_score, N):
    """
    Derive top-N keywords for each live cluster.

    Fixes:
      • skip clusters that still have *zero* term-frequency rows  
      • ensure every item passed to vstack is a csr_matrix  
      • short-circuit early and return empties if no clusters survive  
    """
    start_time = time.time()
    cluster_ids = list(set(window[window["cluster"] >= 0]["cluster"]))

    # collect TF vectors
    cluster_tf_dic = {}
    for cid in cluster_ids:
        if time_aware:
            tf_sum = None
            decaying_factor = window_size
            for date in sorted(cluster_tf_sum_dics[cid].keys())[::-1]:
                delta = (to_date - date).days - 1
                if delta >= window_size:
                    break
                weight = np.exp(-delta / decaying_factor)
                vec = weight * cluster_tf_sum_dics[cid][date]
                tf_sum = vec if tf_sum is None else tf_sum + vec
            # tf_sum might still be None if the cluster is completely empty
            if tf_sum is not None:
                cluster_tf_dic[cid] = tf_sum
        else:
            tf_sum = window[window["cluster"] == cid].article_TF
            if len(tf_sum) > 0:
                cluster_tf_dic[cid] = np.sum(tf_sum)

    # no cluster has any term-frequency data yet
    if not cluster_tf_dic:
        empty = {}, {}, {}, time.time() - start_time
        return empty

    # TF/IDF + BM25 prep
    tf_list = []
    for tf_vec in cluster_tf_dic.values():
        if not isinstance(tf_vec, csr_matrix):
            # convert 1-d np.array or 0-d to 1×V sparse
            tf_vec = csr_matrix(tf_vec)
        tf_list.append(tf_vec)

    cluster_tf = vstack(tf_list)
    cluster_df = np.bincount(cluster_tf.indices,
                             minlength=cluster_tf.shape[1]).reshape(1, -1)
    cluster_idf = np.log((len(tf_list) + 1) / (cluster_df + 1)) + 1

    if keyword_score == "tfidf":
        cluster_keyword_score_all = cluster_tf.multiply(cluster_idf).tocsr()
    elif keyword_score == "bm25":
        k1, b, d = 1.2, 0.75, 1.0
        avgDL = np.sum(cluster_tf) / len(tf_list)
        cluster_ntf = cluster_tf.multiply(
            1 / np.array(1 - b + b * np.sum(cluster_tf, axis=1) / avgDL)
        )
        cluster_ntf.data = ((k1 + 1) * cluster_ntf.data) / (k1 + cluster_ntf.data) + d
        cluster_keyword_score_all = cluster_ntf.multiply(cluster_idf).tocsr()

    # pick top-N tokens
    cluster_topN_indices = {}
    cluster_topN_scores = {}
    cluster_topN_probs = {}

    for i, cid in enumerate(cluster_tf_dic.keys()):
        top_idx = cluster_keyword_score_all[i].indices[
            cluster_keyword_score_all[i].data.argsort()[: -(N + 1) : -1]
        ]
        cluster_topN_indices[cid] = top_idx
        cluster_topN_scores[cid] = cluster_keyword_score_all[i][:, top_idx]

        # compute normalised TF as probability for Jensen-Shannon sims
        top_tf = cluster_tf_dic[cid][:, top_idx]
        probs = (top_tf / np.sum(top_tf)).toarray()[0]
        cluster_topN_probs[cid] = [round(x, 5) for x in probs]

    return (
        cluster_topN_indices,
        cluster_topN_scores,
        cluster_topN_probs,
        time.time() - start_time,
    )

def assign_to_clusters(initial, verbose, window, window_size, to_date, cluster_centers, 
                       cluster_emb_sum_dics, cluster_tf_sum_dics, cluster_topN_probs,
                       T, time_aware = False, theme_aware = False, 
                       cluster_topN_indices = None, cluster_topN_scores = None):
    
    start_time = time.time()

    if initial:
        considered_center_indices = list(range(len(cluster_centers)))
    else:
        considered_center_indices = list(set(window[window['cluster']>=0]['cluster']))

    if verbose: print("Assign to "+str(len(considered_center_indices))+" clusters")
    out_thred = (1-1/(len(considered_center_indices)+1))**T #+1 to handle a single cluster

    if theme_aware:
        sentence_tfs_all = vstack(window[window.cluster==-1]['sentence_TFs'].values)
        article_tfs_all = vstack(window[window.cluster==-1]['article_TF'].values)
        sentence_raw_weights_all = {}
        article_topN_tfs_all = {}
        for cluster_id in considered_center_indices:
            sentence_raw_weights_all[cluster_id] = np.array(np.sum(sentence_tfs_all[:,cluster_topN_indices[cluster_id]].multiply(cluster_topN_scores[cluster_id]), axis=1)).ravel()                       
            article_topN_tfs_all[cluster_id] = article_tfs_all[:,cluster_topN_indices[cluster_id]].toarray()
            
    if time_aware:
        time_weighted_center_dic = {}
        decaying_factor = window_size
    
        for uniq_date in window[window.cluster == -1]["date"].unique():
            for cluster_id in considered_center_indices:
    
                # accumulate embedding sums and weights
                time_weighted_sum = 0
                time_weighted_num = 0
    
                for date, (emb_sum, count) in cluster_emb_sum_dics[cluster_id].items():
                    # skip if outside the window
                    delta = (to_date - date).days - 1
                    if delta >= window_size:
                        continue
    
                    w = np.exp(-abs((uniq_date - date).days) / decaying_factor)
                    time_weighted_sum += w * emb_sum
                    time_weighted_num += w * count
    
                # choose safe centre
                if time_weighted_num == 0:
                    # fallback: old static centre
                    centre = cluster_centers[cluster_id]
                else:
                    centre = time_weighted_sum / time_weighted_num
    
                time_weighted_center_dic[(pd.Timestamp(uniq_date), cluster_id)] = centre

    num_processed_articles = 0
    num_processed_sentences = 0
    for (idx,article) in window[window.cluster==-1].iterrows():
        w_emb = article.embedding # default article embedding
        
        ## Evaluate the similarity to clusters
        if theme_aware:
            similarities = []
            total_weighted_embeddings = []

            for cluster_id in considered_center_indices:                
                sentence_raw_weights = np.array(sentence_raw_weights_all[cluster_id][num_processed_sentences:num_processed_sentences + len(article.sentences)]).ravel()
                if sum(sentence_raw_weights) > 0:
                    sentence_weights = sentence_raw_weights / np.sum(sentence_raw_weights)
                    c_emb = np.matmul(sentence_weights,article.sentence_embds)
                    
                    total_weighted_emb = c_emb
                else: #if any of sentence is weighted, then just use default embedding
                    total_weighted_emb = w_emb 

                total_weighted_embeddings.append(total_weighted_emb)

                if time_aware:
                    time_weighted_center = time_weighted_center_dic[(article['date'], cluster_id)]
                    cos_sim = np.dot(total_weighted_emb, time_weighted_center)/(np.linalg.norm(total_weighted_emb)*np.linalg.norm(time_weighted_center))
                else:
                    cos_sim = np.dot(total_weighted_emb, cluster_centers[int(cluster_id)])/(np.linalg.norm(total_weighted_emb)*np.linalg.norm(cluster_centers[int(cluster_id)]))
                
                if sum(sentence_raw_weights) > 0:
                    article_topN_tfs = article_topN_tfs_all[cluster_id][num_processed_articles]
                    p_cluster  = cluster_topN_probs[cluster_id]
                    p_article = (article_topN_tfs/np.sum(article_topN_tfs))
                    js_sim = 1 - distance.jensenshannon(p_cluster,p_article)
                else:
                    js_sim = 0
                
                if cos_sim < 0: cos_sim = 0
                similarities.append(cos_sim*js_sim)
            num_processed_sentences += len(article.sentences)
            num_processed_articles += 1
        else:
            if time_aware:
                similarities = []
                for cluster_id in considered_center_indices:
                    time_weighted_center = time_weighted_center_dic[(article['date'], cluster_id)]
                    cos_sim = np.dot(article.embedding, time_weighted_center)/(np.linalg.norm(article.embedding)*np.linalg.norm(time_weighted_center))
                    similarities.append(cos_sim)
            else:
                considered_centers = [cluster_centers[int(k)] for k in considered_center_indices]
                similarities = cosine_similarity([article.embedding], considered_centers)
        
        probs = np.exp(T*np.array(similarities)).ravel()
        probs = probs/np.sum(probs)
       
        ## Assign to the most appropriate cluster
        if not initial and len(probs) < 2:
            conf = np.max(similarities) #if a single cluster
        else:
            conf = np.max(probs)
        if 1-conf > out_thred:
            window.at[idx,'cluster'] = -1
            window.at[idx,'sim'] = 0
        else:
            cluster_id = considered_center_indices[np.argmax(probs)]
            window.at[idx,'cluster'] = cluster_id
            window.at[idx,'sim'] = np.max(probs)
            
            if theme_aware: #update embedding
                window.at[idx,'embedding'] = total_weighted_embeddings[np.argmax(probs)]

            if article['date'] not in cluster_emb_sum_dics[cluster_id]:
                cluster_emb_sum_dics[cluster_id][article['date']] = [0,0]
                cluster_tf_sum_dics[cluster_id][article['date']] = 0
            cluster_emb_sum_dics[cluster_id][article['date']][0] += article['embedding'] #embedding sum
            cluster_emb_sum_dics[cluster_id][article['date']][1] += 1 #article count
            cluster_tf_sum_dics[cluster_id][article['date']] += article['article_TF'] #article tf sum

    return window, cluster_emb_sum_dics, cluster_tf_sum_dics, time.time() - start_time

def cluster_outliers(
        window,
        cluster_centers,
        cluster_emb_sum_dics,
        cluster_tf_sum_dics,
        min_articles,
        verbose=False):
    """
    Re-cluster the current outliers (articles with cluster == -1).
    Any newly created cluster must contain at least `min_articles` members.
    Uses spherical k-means implemented via unit-norm + sklearn.KMeans.
    """
    start_time = time.time()

    # locate outliers
    out_idx = window[window["cluster"] == -1].index
    num_new_clusters = int(len(out_idx) / min_articles)

    # conditional recluster
    if num_new_clusters > 1:
        Xo = np.vstack(window.loc[out_idx, "embedding"].values)
        # cosine ⇔ Euclidean after ℓ2-normalisation
        o_labels, o_centers = spherical_kmeans(Xo, num_new_clusters, random_state=0)

        cluster_id_dic, new_centers = {}, []
        next_id = len(cluster_centers)

        # only keep clusters with at least `min_articles` members
        for label in set(o_labels):
            if list(o_labels).count(label) < min_articles:
                continue
            cluster_id_dic[label] = next_id
            new_centers.append(o_centers[label])
            cluster_emb_sum_dics.append({})
            cluster_tf_sum_dics.append({})
            if verbose:
                size = list(o_labels).count(label)
                print(f"A new cluster {next_id} of {size} articles is created")
            next_id += 1

        # update assignments
        for i, art_idx in enumerate(out_idx):
            lbl = o_labels[i]
            if lbl in cluster_id_dic:
                cid = cluster_id_dic[lbl]
                window.at[art_idx, "cluster"] = cid
                article_date = window.at[art_idx, "date"]

                cluster_emb_sum_dics[cid].setdefault(article_date, [0, 0])
                cluster_tf_sum_dics[cid].setdefault(article_date, 0)

                # update running sums
                cluster_emb_sum_dics[cid][article_date][0] += window.at[art_idx, "embedding"]
                cluster_emb_sum_dics[cid][article_date][1] += 1
                cluster_tf_sum_dics[cid][article_date] += window.at[art_idx, "article_TF"]
            else:
                # remains an outlier
                window.at[art_idx, "cluster"] = -1

        # append the new centroids (already unit-normalised)
        cluster_centers = np.array(list(cluster_centers) + new_centers)

    # return unchanged API
    return (
        window,
        cluster_centers,
        cluster_emb_sum_dics,
        cluster_tf_sum_dics,
        time.time() - start_time,
    )

def update_cluster_keywords_articles(i, window, all_vocab, cluster_keywords_df, cluster_topN_indices):
    for k in cluster_topN_indices.keys(): 
        if k not in cluster_keywords_df.columns:
            cluster_keywords_df[k] = ''
            cluster_keywords_df[k] = cluster_keywords_df[k].astype('object')
        cluster_keywords_df.at[i,k] = ''
        cluster_keywords_df.at[i,k] = [all_vocab[i] for i in cluster_topN_indices[k]]
    return cluster_keywords_df

# ------------------------------------------------------------------------------
# script to run from cold-start or test
# ------------------------------------------------------------------------------

def simulate(
    file_path,
    window_size,
    slide_size,
    num_windows,
    min_articles,
    N,
    T,
    keyword_score,
    verbose,
    story_label,
    time_aware=True,
    theme_aware=True,
):

    article_df, all_vocab = read_dataset(file_path, story_label, verbose)
    begin_date = article_df.date.iloc[0].strftime("%Y-%m-%d")

    all_window, window, cluster_keywords_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    window_indices, article_df_slides = [], []
    eval_metrics, win_proc_times = [], []

    article_df["cluster"] = -1
    article_df["sim"] = 0.0
    cluster_centers = []

    for i in range(num_windows):
        init_start_time = time.time()
        if verbose:
            print(f"<Window {i}>")

        # [1] build slide -------------------------------------------------------
        from_date = pd.to_datetime(begin_date) + pd.DateOffset(days=i * slide_size)
        to_date = pd.to_datetime(begin_date) + pd.DateOffset(days=(i + 1) * slide_size)
        slide = article_df[
            (article_df["date"] >= from_date) & (article_df["date"] < to_date)
        ].copy()

        if len(window_indices) >= window_size / slide_size:
            all_window = pd.concat([all_window, window.loc[window_indices[0]]])
            window.drop(index=window_indices[0], inplace=True)
            window_indices.pop(0)
            article_df_slides.pop(0)

        if len(slide) < 1:
            article_df_slide = np.zeros(len(all_vocab)).reshape(1, -1)
        else:
            article_TFs = vstack(slide["article_TF"])
            article_df_slide = np.bincount(
                article_TFs.indices, minlength=article_TFs.shape[1]
            ).reshape(1, -1)

        window_indices.append(slide.index)
        article_df_slides.append(article_df_slide)

        # embeddings -----------------------------------------------------------
        slide, _ = get_article_embedding(
            slide,
            window,
            article_df_slides,
            time_aware,
            theme_aware,
            keyword_score,
            N,
        )
        window = pd.concat([window, slide])

        # [2] initialise clusters ---------------------------------------------
        if len(cluster_centers) == 0:
            num_new_clusters = int(len(window) / min_articles)
            if num_new_clusters < 1:
                continue
            X = np.vstack(window["embedding"].values)
            labels, centres = spherical_kmeans(X, num_new_clusters, random_state=0)
            cluster_centers = centres.tolist()
            if verbose:
                print(f"{len(cluster_centers)} clusters are initialized")

            cluster_emb_sum_dics = [{} for _ in range(len(cluster_centers))]
            cluster_tf_sum_dics = [{} for _ in range(len(cluster_centers))]
            cluster_topN_probs = {}

            initial = True
            window, cluster_emb_sum_dics, cluster_tf_sum_dics, _ = assign_to_clusters(
                initial,
                verbose,
                window,
                window_size,
                to_date,
                cluster_centers,
                cluster_emb_sum_dics,
                cluster_tf_sum_dics,
                cluster_topN_probs,
                T,
            )

        # [2b] assign new points after init ------------------------------------
        elif (
            len(set(window[window["cluster"] >= 0]["cluster"])) > 0
            and len(window[window["cluster"] == -1]) > 0
        ):
            initial = False
            window, cluster_emb_sum_dics, cluster_tf_sum_dics, _ = assign_to_clusters(
                initial,
                verbose,
                window,
                window_size,
                to_date,
                cluster_centers,
                cluster_emb_sum_dics,
                cluster_tf_sum_dics,
                cluster_topN_probs,
                T,
                time_aware,
                theme_aware,
                cluster_topN_indices,
                cluster_topN_scores,
            )

        # [3] cluster outliers --------------------------------------------------
        window, cluster_centers, cluster_emb_sum_dics, cluster_tf_sum_dics, _ = cluster_outliers(
            window,
            cluster_centers,
            cluster_emb_sum_dics,
            cluster_tf_sum_dics,
            min_articles,
            verbose,
        )

        # [4] theme keywords ----------------------------------------------------
        if len(set(window[window["cluster"] >= 0]["cluster"])) > 0:
            (
                cluster_topN_indices,
                cluster_topN_scores,
                cluster_topN_probs,
                _,
            ) = get_cluster_theme(
                window,
                window_size,
                to_date,
                time_aware,
                cluster_tf_sum_dics,
                keyword_score,
                N,
            )

        # stats -----------------------------------------------------------------
        if len(window) > 0:
            win_proc_times.append(time.time() - init_start_time)
            cluster_keywords_df = update_cluster_keywords_articles(
                i, window, all_vocab, cluster_keywords_df, cluster_topN_indices
            )

    all_window = pd.concat([all_window, window])

    if story_label:
        nmi, ami, ri, ari, precision, recall, fscore = [
            np.round(k, 3) for k in np.mean(eval_metrics, axis=0)
        ]
    else:
        nmi, ami, ri, ari, precision, recall, fscore = [0] * 7

    final_num_cluster = len(cluster_centers)
    avg_win_proc_time = np.round(np.mean(win_proc_times), 1)

    return (
        all_window,
        cluster_keywords_df,
        final_num_cluster,
        avg_win_proc_time,
        nmi,
        ami,
        ri,
        ari,
        precision,
        recall,
        fscore,
    )
