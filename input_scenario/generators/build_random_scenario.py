import json
import random
import re

# 기존 JSON 데이터
scenarios_data = {
    "1_founder_period": {
      "problem_generic": "Problem: A new company must select an initial market entry strategy to prove its concept and secure future funding, despite lacking a proven supply chain, manufacturing assets, and brand recognition. The core challenge is to create a compelling and differentiated narrative while managing a high capital risk.",
      "problem_specific": "Problem: Tesla, a new company, must select an initial market entry strategy to prove its concept and secure future funding, despite lacking a proven supply chain, manufacturing assets, and brand recognition. The core challenge is to create a compelling and differentiated narrative while managing a high capital risk.",
      "context_blocks": {
        "[Market] Global new car sales: ~60 million units/year.": "Global new car sales were roughly ~60 million units per year.",
        "[Market] EV share: <0.02% of global new car sales.": "Battery electric vehicles accounted for less than ~0.02% of global new car sales.",
        "[Technology] Li-ion cell cost: ~1,000 USD/kWh.": "Lithium-ion cell costs were around ~1,000 USD per kWh.",
        "[Technology] Prototype performance: 0–60 mph in 3.6 s; ~300 km range.": "A high-performance prototype achieved 0–60 mph in ~3.6 seconds with ~300 km range, demonstrating the performance potential of EVs.",
        "[Finance] Initial funding: ~USD 7.5M Series A (private).": "Initial external funding amounted to approximately USD 7.5 million in a Series A round, indicating limited capital.",
        "[Market] Incumbent focus: Established automakers concentrated on hybrid and fuel cell technologies, not pure EVs.": "Major automakers at the time were focused on hybrid and fuel cell technologies, not pure electric vehicles."
      },
      "execution_options": {
        "A": { "name": "Build a portfolio of hybrid/fuel cell vehicles", "mapping": "Fast Follower" },
        "B": { "name": "Develop a high-performance EV sports car", "mapping": "Technology Leadership" },
        "C": { "name": "Co-develop an EV platform with a major automaker", "mapping": "Open Innovation" }
      }
    },
    "2_roadster_launch": {
      "problem_generic": "Problem: A company must manage conflicting goals of product quality and timely delivery during its initial product launch. With significant pre-orders already placed, the company faces severe cash flow issues and supply chain delays, jeopardizing brand trust and future investment if not handled correctly.",
      "problem_specific": "Problem: Tesla must manage conflicting goals of product quality and timely delivery during its initial product launch. With significant pre-orders already placed, the company faces severe cash flow issues and supply chain delays, jeopardizing brand trust and future investment if not handled correctly.",
      "context_blocks": {
        "[Customer Response] Initial pre-orders: over 900 units.": "Initial pre-orders amounted to over 900 units, indicating high customer interest.",
        "[Market] Production ramp-up: 27-70 units delivered in the first months of production.": "Initial deliveries in the first months of production were very low, ranging from 27 to 70 units.",
        "[Finance] Investment raised: $40M secured for production start and dealer network.": "A $40M investment was secured to begin production and build a dealer network, but cash flow remained tight.",
        "[Manufacturing] Supply Chain: Delays from key partners and frequent design changes hindered production.": "Frequent supply delays from key partners and design changes became a major bottleneck."
      },
      "execution_options": {
        "A": { "name": "Prioritize product quality and performance, accepting launch delays", "mapping": "Technology Leadership" },
        "B": { "name": "Accelerate launch to meet demand, accepting potential quality compromises", "mapping": "Fast Follower" },
        "C": { "name": "Expand manufacturing partnerships to share risk", "mapping": "Open Innovation" }
      }
    },
    "3_model_s_launch": {
      "problem_generic": "Problem: A company needs to transition from a niche-market player to a mass-market manufacturer by establishing a large-scale production infrastructure. The key is to secure trust in the premium sedan market and build a production base for eventual expansion into lower-priced models.",
      "problem_specific": "Problem: Tesla needs to transition from a niche-market player to a mass-market manufacturer by establishing a large-scale production infrastructure. The key is to secure trust in the premium sedan market and build a production base for eventual expansion into lower-priced models.",
      "context_blocks": {
        "[Finance] Large factory acquisition: $42M.": "The company acquired a large manufacturing factory for $42M, securing a large production facility.",
        "[Policy/Regulation] Government loan: $465M support.": "The company received a $465M government loan, providing significant financial backing.",
        "[Market] Fuel cost savings: A new premium sedan was projected to save about $1,800 over 6 years.": "A new premium sedan was projected to save customers about $1,800 in fuel costs over 6 years, a strong selling point for consumers."
      },
      "execution_options": {
        "A": { "name": "Focus exclusively on the premium EV sedan market", "mapping": "Technology Leadership" },
        "B": { "name": "Immediately attempt to launch a lower-priced model to popularize EVs", "mapping": "Fast Follower" },
        "C": { "name": "Sell battery packs and powertrains to other companies", "mapping": "Open Innovation" }
      }
    },
    "4_model_x_launch": {
      "problem_generic": "Problem: A company aims to enter the growing SUV market. However, a complex product design creates high production difficulty and quality risks, which could severely damage the brand's reputation despite a lack of direct competition.",
      "problem_specific": "Problem: Tesla aims to enter the growing SUV market. However, a complex product design creates high production difficulty and quality risks, which could severely damage the brand's reputation despite a lack of direct competition.",
      "context_blocks": {
        "[Market] US SUV market share: about 32% of new car sales.": "SUVs made up about 32% of new car sales in the US, a large and growing segment.",
        "[Market] Competition: There were very few competing electric SUVs at the time.": "At the time, there were very few competing electric SUVs.",
        "[Technology] Design Complexity: A new design with complex, upward-swinging doors posed significant manufacturing challenges.": "A new design with complex, upward-swinging doors posed significant manufacturing challenges.",
        "[Customer Response] High pre-order volume: Over 30,000 pre-orders existed, increasing pressure to deliver.": "There was a high pre-order volume, increasing the pressure to deliver."
      },
      "execution_options": {
        "A": { "name": "Launch a luxury SUV with innovative, complex features", "mapping": "Technology Leadership" },
        "B": { "name": "Develop a standard, mid-priced SUV to enter the mass market", "mapping": "Niche Focus" },
        "C": { "name": "Postpone the launch to stabilize existing product production", "mapping": "Maintain" }
      }
    },
    "5_model_3_mass_market": {
      "problem_generic": "Problem: Facing an explosive increase in pre-orders, a company must rapidly scale production while maintaining product quality, financial stability, and public trust. The core challenge is to overcome production bottlenecks, dubbed 'Production Hell', and reduce costs without sacrificing the brand’s reputation in the mass market.",
      "problem_specific": "Problem: Tesla must rapidly scale production while maintaining product quality, financial stability, and public trust. The core challenge is to overcome production bottlenecks, dubbed 'Production Hell', and reduce costs without sacrificing the brand’s reputation in the mass market.",
      "context_blocks": {
        "[Customer Response] First week pre-orders: over 325,000 units.": "Over 325,000 units were pre-ordered in the first week.",
        "[Finance] Value of orders: about $14B.": "The value of pre-orders was approximately $14 billion.",
        "[Customer Response] Net pre-orders after cancellations: 373,000.": "Net pre-orders after removing cancellations and duplicates were 373,000.",
        "[Market] Global EV sales and company production volume are surging.": "Global EV sales and the company's production volume are surging, indicating a rapid market shift."
      },
      "execution_options": {
        "A": { "name": "Prioritize production speed to meet demand as quickly as possible", "mapping": "Technology Leadership" },
        "B": { "name": "Expand production gradually while prioritizing quality and profitability", "mapping": "Maintain" },
        "C": { "name": "Utilize manufacturing partners (OEM) to scale production", "mapping": "Open Innovation" }
      }
    },
    "6_energy_infra": {
      "problem_generic": "Problem: A company needs to address key bottlenecks for mass EV adoption: the lack of charging infrastructure and high battery costs. The strategic goal is to diversify revenue beyond the core automotive business by integrating energy storage and generation to create synergy.",
      "problem_specific": "Problem: Tesla needs to address key bottlenecks for mass EV adoption: the lack of charging infrastructure and high battery costs. The strategic goal is to diversify revenue beyond the core automotive business by integrating energy storage and generation to create synergy.",
      "context_blocks": {
        "[Technology] Proprietary charging network: over 2,000 stations, 18,000+ chargers (2020).": "The company's proprietary charging network had over 2,000 stations and 18,000+ chargers by 2020, demonstrating a significant investment in infrastructure.",
        "[Technology] Large-scale factory production capacity: 35 GWh per year.": "The company's large-scale factory had a production capacity of 35 GWh per year, a key asset for battery supply.",
        "[Market] Solar module price: fell by about 80% over 10 years.": "Solar module prices fell by about 80% over a decade, making solar energy more viable.",
        "[Strategy] Competitors' Infrastructure: Most other automakers relied on third-party charging networks, not building their own.": "Competitors typically relied on third-party charging networks, not proprietary ones."
      },
      "execution_options": {
        "A": { "name": "Build a proprietary charging network", "mapping": "Technology Leadership" },
        "B": { "name": "Construct a large-scale, proprietary battery factory", "mapping": "Technology Leadership" },
        "C": { "name": "Aggressively expand into energy storage and solar products", "mapping": "Diversification" },
        "D": { "name": "Focus exclusively on the core automotive business", "mapping": "Retrenchment" }
      }
    }
}

import json
import random
import re

# 기존 JSON 데이터
# scenarios_data = ... (위와 동일한 데이터)

def randomize_numbers_in_context(data, variability=0.20):
    """
    context_blocks 내의 숫자들을 일관성 있게 무작위로 변경하는 함수
    
    Args:
        data (dict): JSON 파일에서 불러온 시나리오 데이터
        variability (float): 원래 값에서 변경될 수 있는 최대 비율 (예: 0.20 = ±20%)
        
    Returns:
        dict: 숫자가 변경된 새로운 데이터 딕셔너리
    """
    randomized_data = data.copy()
    
    for scenario_name, scenario_content in randomized_data.items():
        if "context_blocks" in scenario_content:
            new_context_blocks = {}
            for key, value in scenario_content["context_blocks"].items():
                
                # 키에서 숫자 찾기
                numbers_in_key = re.findall(r'(\d+\.?\d*)', key)
                
                new_key = key
                new_value = value
                
                # 키에 있는 숫자 변경 및 값에 동일하게 적용
                if numbers_in_key:
                    for num_str in numbers_in_key:
                        original_num = float(num_str)
                        if original_num != 0:
                            # 0이 아닌 경우에만 변경
                            change = original_num * variability
                            new_num = round(random.uniform(original_num - change, original_num + change), 2)
                            
                            # 키와 값 모두에 동일한 새 값으로 대체
                            new_key = new_key.replace(num_str, str(new_num), 1)
                            new_value = new_value.replace(num_str, str(new_num), 1)
                
                new_context_blocks[new_key] = new_value
            
            randomized_data[scenario_name]["context_blocks"] = new_context_blocks
            
    return randomized_data

# 함수 실행
randomized_scenarios_data = randomize_numbers_in_context(scenarios_data)

# 새로운 JSON 파일로 저장
with open("scenarios_randomized_numbers.json", "w", encoding="utf-8") as f:
    json.dump(randomized_scenarios_data, f, indent=2, ensure_ascii=False)
    
print("숫자가 일관성 있게 무작위로 변경된 'scenarios_randomized_numbers.json' 파일이 생성되었습니다.")