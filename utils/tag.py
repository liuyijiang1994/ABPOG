import common


def get_tags(path, tag_map=common.tag_dict):
    begin_tag = tag_map.get("B")
    mid_tag = tag_map.get("I")
    o_tag = tag_map.get("O")
    end_tag = tag_map.get('E')
    begin = -1
    tags = []
    last_tag = -1
    for index, tag in enumerate(path):
        if tag == begin_tag:
            # 单B
            if index == len(path) - 1:
                end = index
                tags.append([index, end])
            # IB/BB/EB
            if last_tag in [mid_tag, begin_tag, end_tag] and begin > -1:
                end = index - 1
                tags.append([begin, end])
            begin = index
        elif tag == end_tag and last_tag in [mid_tag, begin_tag] and begin > -1:
            end = index
            tags.append([begin, end])
            begin = -1
        elif tag == o_tag and last_tag in [begin_tag, end_tag] and begin > -1:
            end = index - 1
            tags.append([begin, end])
            begin = -1
        elif tag == o_tag:
            begin = -1
        last_tag = tag
    return tags
