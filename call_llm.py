def get_prompt_generate(history, field):
    f = field.strip('s')
    prompt = '['+ ', '.join(history) + ']'
    return prompt


def get_system_generate(history, field):
    f = field.strip('s')
    return f"""
Suppose you are an interest inferrer. Your function is to infer the user's interests from the given list. You will be provided with a list of {field} an anonymous user has liked, and your task is to infer the user's interests based on the list and your extensive knowledge. List no more than five the top interests of this anonymous user by analyzing recurring themes, genres, and emotional tones across the items. No further explanation is needed. Please use a comma to split the interests. Exclude generic terms like "fiction" unless dominant.
"""

