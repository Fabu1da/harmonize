import dotenv
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull

from openai import OpenAI
import numpy as np

dotenv.load_dotenv()
client = OpenAI()

def embed(text, model="text-embedding-3-small"):
    out = client.embeddings.create(model=model, input=text)
    return np.array(out.data[0].embedding, dtype=float)

a = "spouse"
b = "partner"
c = "father"
d = "wife"
e = "husband"
f = "mother"
g = "brother"
h = "sister"
i = "son"
j = "daughter"
k = "grandfather"
l = "grandmother"
m = "uncle"
n = "aunt"
o = "cousin"
p = "nephew"
q = "niece"
r = "friend"
s = "colleague"
t = "neighbor"

ea = embed(a)
eb = embed(b)
ec = embed(c)
ed = embed(d)
ef = embed(e)
eg = embed(f)
eh = embed(g)
ei = embed(h)
ej = embed(i)
ek = embed(j)
el = embed(k)
em = embed(l)
en = embed(m)
eo = embed(n)
ep = embed(o)
eq = embed(p)
er = embed(q)
es = embed(r)
et = embed(s)
eu = embed(t)

words = ["spouse","partner","father","wife", "mother", "brother", "sister", "son", "daughter", "grandfather", "grandmother", "uncle", "aunt", "cousin", "nephew", "niece", "friend", "colleague", "neighbor"]
vecs = np.array([ea, eb, ec, ed, ef, eg, eh, ei, ej, ek, el, em, en, eo, ep, eq, er, es, et, eu])

def cos(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print("colleague vs spouse:", cos(eg, ea))
print("neighbor vs spouse:", cos(et, ea))
print("colleague vs neighbor:", cos(eg, et))


pca = PCA(n_components=2)
points = pca.fit_transform(vecs)

# Perform clustering
n_clusters = 4  # Adjust as needed
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(points)

# Create color map for clusters
colors = plt.cm.Set1(np.linspace(0, 1, n_clusters))

# Plot points with cluster colors
for cluster_id in range(n_clusters):
    cluster_points = points[clusters == cluster_id]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                c=[colors[cluster_id]], label=f'Cluster {cluster_id}', alpha=0.7)
    
    # Draw convex hull border for each cluster
    if len(cluster_points) >= 3:
        hull = ConvexHull(cluster_points)
        for simplex in hull.simplices:
            plt.plot(cluster_points[simplex, 0], cluster_points[simplex, 1], 
                     color=colors[cluster_id], linewidth=2)

# Add word labels
for i, w in enumerate(words):
    plt.text(points[i,0]+0.01, points[i,1]+0.01, w)

plt.legend()
plt.show()
