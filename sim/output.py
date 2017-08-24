import json, os

def output_json(stress_matrix, points, clusters):
    json_file = open('www/static/stress.json', 'w')

    out = {
        "type": "NetworkGraph",
        "label": "Topic Stress Graph",
        "protocol": "MDS",
        "version": "0.0.1v",
        "metric": "Stress"
    }
    nodes = []
    links = []

    N = len(stress_matrix)
    for i in range(N):
        node = {'id': i, 'x': points[i][0].item(), 'y': points[i][1].item(), 'cluster': clusters[i].item()}
        nodes.append(node)
        for j in range(N):
            if i > j:
                dist = stress_matrix[i][j]
                #link = {"source": i, "target": j, "cost": dist}
                link = {'x1': points[i][0].item(), 'y1': points[i][1].item(), 'x2': points[j][0].item(), 'y2': points[j][1].item(), 'cost': dist}
                links.append(link)

    out['nodes'] = nodes
    out['links'] = links

    json.dump(out, json_file)
