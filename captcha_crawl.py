from pyquery import PyQuery as pq
import urllib2

def curl(url):
    """
    return content at url.
    return empty string if response raise an HTTPError (not found, 500...)
    """
    try:
        print "retrieving url... %s" % (url)
        req = urllib2.Request(url)
        proxy = urllib2.ProxyHandler({'http': '10.50.28.2:3128'})
        opener = urllib2.build_opener(proxy)
        urllib2.install_opener(opener)
        response = urllib2.urlopen(req)
        return response.read().decode('ascii', 'ignore')
    except urllib2.HTTPError, e:
        print "error %s: %s" % (url, e)
        return ''

def get_url():
    base_url = "http://sistemas3.sef.sc.gov.br/sintegra/"
    first_context = "consulta_empresa_pesquisa.aspx"
    response = curl(base_url+first_context)
    d = pq(response)
    img = d("#UpdatePanel1")("img")
    img_context = img.attr.src
    return base_url + img_context
