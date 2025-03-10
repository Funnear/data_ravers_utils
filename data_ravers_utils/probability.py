import math

# to test the import
def hello():
    print("hello")

class OneDimensionalSet:
    def __init__(self,  item_type: str):
        self.item_type = item_type
        self.subsets = dict()
    
    def add_subset(self, subtype:str, count: int):
        self.subsets[subtype] = count

    def get_subsets(self, notebook_code=False):
        if notebook_code:
            code_string_list = list()

            for key, value in self.subsets.items():
                code_string_list.append(f"{key}_{self.item_type}_count = {value}\n")
            
            code_string_list.append("\n")
            return "".join(code_string_list)
        else:
            return self.subsets
    
    def get_total_count(self, notebook_code=False):
        if notebook_code:
            code_string_list = list()

            code_string_list.append(f"total_{self.item_type} = ")
            for key in self.subsets:
                code_string_list.append(f"+ {key}_{self.item_type}_count")
            code_string_list.append("\n")

            code_string_list.append("\n")
            return "".join(code_string_list)
        else:
            return sum(self.subsets.values())
        
    def get_item_count(self):
        return len(self.subsets)
    
    def prob_all_unique(self, notebook_code=False):
        if notebook_code:
            code_string_list = list()
            
            code_string_list.append(self.get_subsets(notebook_code=True))
            code_string_list.append(self.get_total_count(notebook_code=True))

            code_string_list.append("numerator = (")
            for key in self.subsets:
                code_string_list.append(f"* {key}_{self.item_type}_count")
            code_string_list.append(")\n")
            
            code_string_list.append("denominator = (")
            code_string_list.append(f"total_{self.item_type}")
            for i in range(1, self.get_item_count()):
                code_string_list.append(f"* (total_{self.item_type}-{i})")
            code_string_list.append(")\n")

            code_string_list.append("quotient = numerator / denominator\n")
            code_string_list.append("print(quotient)\n")

            # (n_white_balls * n_red_balls * n_black_balls)/((total_balls) * (total_balls-1) * (total_balls-2))
            code_string_list.append("\n")
            return "".join(code_string_list)
        else:
            self.total_count = self.get_total_count()
            numerator = math.prod(self.subsets.values())
            denominator = 1

            for i in range(self.get_item_count()):
                denominator *= self.total_count -i

            quotient = numerator / denominator
            return quotient
