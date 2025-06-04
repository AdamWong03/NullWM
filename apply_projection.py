# null-space projection hook function placeholder
def project_hidden(h, V_k):
    return h - (h @ V_k.T) @ V_k
