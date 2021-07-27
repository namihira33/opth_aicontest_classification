def iterate(d, param={}):
    d, param = d.copy(), param.copy()
    d_list = []

    for k, v in d.items():
        if isinstance(v, list):
            for vi in v:
                d[k], param[k] = vi, vi
                d_list += iterate(d, param)
            return d_list

        if isinstance(v, dict):
            add_d_list = iterate(v, param)
            if len(add_d_list) > 1:
                for vi, pi in add_d_list:
                    d[k] = vi
                    d_list += iterate(d, pi)
                return d_list

    return [[d, param]]