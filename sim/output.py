import json, pickle, os
import sim.topics
import numpy as np
from yattag import Doc

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
        node = {'id': i, 'fx': points[i][0].item(), 'fy': points[i][1].item(), 'cluster': clusters[i].item()}
        nodes.append(node)
        for j in range(N):
            if i > j:
                dist = stress_matrix[i][j]
                link = {"source": i, "target": j, "cost": dist}
                #link = {'x1': points[i][0].item(), 'y1': points[i][1].item(), 'x2': points[j][0].item(), 'y2': points[j][1].item(), 'cost': dist}
                links.append(link)

    out['nodes'] = nodes
    out['links'] = links

    json.dump(out, json_file)
    json_file.close()

def output_html_list(stress_points, total_stress_points, quads):
    N = len(stress_points)

    doc, tag, text = Doc().tagtext()

    doc.asis('<!DOCTYPE html>')
    with tag('html'):
        with tag('head'):
            doc.stag('link', rel='stylesheet', href="{{ url_for('static', filename='reset.css') }}")
            doc.stag('link', rel='stylesheet', href="{{ url_for('static', filename='stress-list.css') }}")
        with tag('body'):
            iquad = 1
            for quad in quads:
                with tag('h2'):
                    text(iquad)
                    with tag('ul'):
                        for point_stress in quad:
                            point = point_stress[0]
                            stress = '{0:.5f}'.format(point_stress[1])
                            with tag('li'):
                                text('%s: %s' % (point, stress))
                iquad += 1
            with tag('h2'):
                text('Total Stress per Point')
                with tag('ul'):
                    for total_stress_point in total_stress_points:
                        point = total_stress_point[0]
                        stress = '{0:.5f}'.format(total_stress_point[1])
                        with tag('li'):
                            text('%s: %s' % (point, stress))
            with tag('h2'):
                text('Stress points')
                with tag('ul'):
                    for stress_point in stress_points:
                        point = stress_point[0]
                        stress = '{0:.9f}'.format(stress_point[1])
                        with tag('li'):
                            text('%s | %s' % (point, stress))

    html_file = open('www/templates/stress.html', 'w')
    html_file.write(doc.getvalue())
    html_file.close()

# move logic to flask
def output_html(stress_matrix, points, clusters):
    N = len(stress_matrix)

    doc, tag, text = Doc().tagtext()

    doc.asis('<!DOCTYPE html>')
    with tag('html'):
        with tag('head'):
            doc.stag('link', rel='stylesheet', href="{{ url_for('static', filename='reset.css') }}")
            doc.stag('link', rel='stylesheet', href="{{ url_for('static', filename='stress.css') }}")
        with tag('body'):
            with tag('table'):
                with tag('tr'):
                    with tag('th'):
                        text(' ')
                    for i in range(N):
                        with tag('th'):
                            text(i)
                for i in range(N):
                    with tag('tr'):
                        with tag('th'):
                            text(i)
                        for j in range(N):
                            stress = stress_matrix[i][j]
                            with tag('td', title=stress):
                                text('{0:.2f}'.format(stress))

    html_file = open('www/templates/stress.html', 'w')
    html_file.write(doc.getvalue())
    html_file.close()
