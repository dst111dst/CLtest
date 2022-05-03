import pickle

class FeatureBasedSimilarity:
    def __init__(self,similarity_path):
        """
        :param : similarity_path -> the trained similarity model path
        """
        self.similarity_path = similarity_path
        with open(similarity_path, 'rb') as read_file:
            self.similarity_dict = pickle.load(read_file)

    def most_similar(self, item, top_k=10):
        if top_k > 10:
            top_k = 10
        if isinstance(item,str):
            item = int(item)
        try:
            top_k_items=self.similarity_dict[item]
            return top_k_items[0:top_k]
        except Exception as e:
            print(e)
            return None



if __name__ == "__main__":
    sv_path = '/Users/tt/Downloads/cl4pps/src/models/similarity.pkl'
    feature_based_similarity = FeatureBasedSimilarity(similarity_path=sv_path)
    print(feature_based_similarity.most_similar('2', top_k=4))