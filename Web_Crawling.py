# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 22:53:43 2023

@author: Jacob
"""

import numpy as np
import pandas as pd
import requests as req
import json
import time


def web_crawling_lvr_land(url, headers):
    try:
        resp = req.get(url, headers)
        resp.raise_for_status()
        data = json.loads(resp.text)
    except Exception as err:
        print(err)
    df = pd.DataFrame(data)   
    return df

def create_district_name(dataframe, district_name):
    dataframe['district'] = district_name
    return dataframe

header={
  'User-Agent':'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/34.0.1847.137 Safari/537.36 LBBROWSER'
  }
district_name_list = ['東區', '西區', '南區', '北區', '中區', '西屯區', '北屯區', '南屯區']

url_east = 'https://lvr.land.moi.gov.tw/SERVICE/QueryPrice/41b610066005dda3051ed9aa1632faaf?q=VTJGc2RHVmtYMStkSDJqclhUVmdYLzZIYWFnVVlGQ3JyRVJ6TDgxS0N2anUrVDFqUzlsZE1ROGJtTU81d1J6Ulp6Nm45M1llUG5jR3hpM1VkbjgxTFFBVDNkZ2NpWnkwSFFDVFdkVWRwbkxFd1lkakJJanFvZ1Uxb3JZVzF0d29NYW1heFFNOGIyZStOdXNyN0dNWWMyVGk4VTEydU05aHhMamh3bTBwRmcvWGNBMmlJOWJiVlpqS1VqM2g0Q3pVcmF0ZWdhQSt3WUUxeUszMjVCQTNFQS9lR21GWUwvOG8wN2VHanJkVHNBNmJaS29OUUNldncrS1RPT3JPdU9XYlpvSU9UNmtRTytna0J6aWNJY1F0bDV1NGhEakd1RUtRVGRwVmRVK1hxQ0FVRW1USnN3VVJabzMvbEtQQnhkRHJhcmljTlloNWtWOUhWVmhONHNxVGEzSVBKbHh6N3BmczBZMWt6ZjAwL3A3ejY5M09IUEp1YUJHK09icFNETm1lRVVhOEFMb3RodE1DM3JXbDFmOVB1enNiNkJmWWZuWWJTMXRiOEYwT0wzWWJZYnFlMjUwZEJDeGsyMWRTZ0dWVTBmU3JKU0VHa2c4K1UrRTJnN0FRM2ZYdnFQcTh1bHBBTzJsbHdRTko5YzBPSWUxcE92aXFCTHhHMS9QUnFjSVdNa2JjSmFBTHQrNEhKRzRFR0VnS3NtNHVidG13NVl5NDZaYWFrU0J4NXlTbG1TMjNQaUFaWnJvOWtNL2ZUN0UzMlFudXlFcklCbkg0NnhEUnBDbVZ5UkxYeW5CbWVGWGttRlR6T0lZNE1pY2x3SVRncVlqQ1JUMXNoYzZremxFdGVLUi9IRkxwNDhhUkRsbzE0MjNSSHBtWXFIdkNVUkw1WFVITm1tMXdXTEk9'
url_west = 'https://lvr.land.moi.gov.tw/SERVICE/QueryPrice/1ef0fc4c81f84a125831c5b50c348b0b?q=VTJGc2RHVmtYMS9lakw4dU4zOWVRMnR1YzFkSSs5SGEydmwxbjdEbUU5N2I3dXk4RFlGaXNlSnhKNUJOcWxqVTlHaFRqRnI5cU91Q0NndGVMZDNZSXR2TkxKQnZ6WjJHTHlLekc4SEZFVDlaUTliaFd5dDFYMHFVR28xNHdqSzFjcmNyRzUzS2kwOWdlYi93Q2xQMW4xeFVFUFNSSXc5TTNSSDJCRjEzUWJXRVZ5TWUwVlBSeVlmVXBBREdJcGNFUVhQL3Z3b3RqNGxVbmlSWFVmMUJ3eTYyZDZKRUxaZy8rcUpydDhIRDNZU0hYWU9LVkNtMGU0N1hjNEl0RzhLWDhRUUtVRjA2ZnZ3K3hYZnVubnhJL0syb2ZKVTBYVXpJNkd2b05JMXhZdHNGVFdja3lvcldaZ0l0VGlEUEdnbEUyNEI1RnUxVTFTZHZoUS9HUWxIRDVaUFd1TndWV0cwQ3krcENLVC9GRm5VdUlyVGZNWVlJYzJJVjBJazZhdHdTV1FacnNlWi9uYWhXWWQzQWgrN01DejhTOElxdzg4c0I4MEljYmkyRWo1VjRmREJDS1g5NENOcUZCVVRoK1dQcTJ4YnpQR29qR1owY0JoWnFMYzQ2RXB5TFVvYUVDRTBPQThZbzlIMWRhWDdVc1lDR3FDdFNLS3ZHak9MS241OUt5UGRWTDlYNTQzUFN6Ujk2aUxMS1VhYTZ5ZlFreVZFbzVuMzhhK2RtelRCOG5KQ3RkczE0V0YwaGx4aXFXWTZNM0ZVVlFHTmYwUHFTVDZ0dzhnZThIUmd2YWNLdkRUcmVxWkVib3U4VVpJT2lmK01LY1c0OE9BODk2MVd2a0VQQzF1eDNydmw4YTVxU01iek00eVRNUjN1TXNVTmtCQk5zQ3BkWU8wQm1wbzA9'
url_south = 'https://lvr.land.moi.gov.tw/SERVICE/QueryPrice/f0b0e42d27f1f722626a3a514ad0ded9?q=VTJGc2RHVmtYMThoNFBYU202cWxCSDNaTlVnZVIxZ3Q4TjlXM3lZanFkcEFXZFRkam0rNlBpbDNiVnFoWUtCMHNHOXY4bXl6R0xDb3lWd0VpVWJJV0J3TllkK2FPNEE1bk5kMFdKalhtVWRjdmpVOGpqbEloQVRwS1A2RURaMUtYRmRqa2tiYlBLeHJaeWZtTWh2MDl1TG1TNm5sdysvOEQ0R2ZhNHVwS2JFaUdldGg3RWNaWTZaaUVza2ViclJSQUU5YnNRdzl3bThvSG42dzVKTCtvZFE5dkpGeEw1aTFnVTZkNFV3L25LUktwS05OYjVaUTFJdWhYOHVmL0syY3FudlFDQlQzeksyaVcrU2dYR2s1N29mVVIvZmc1djVOc25qbGpSazNhQmNOY2dtVWNzT2tEakhLUmF2RmdUYWI1bmFldEphNUFZKzlPb3FpdlNlM3hTVXYzcFQ5K2RYZ0pDYUFaanUvUHhycG93c0tRKys5QkdtZmNtQVFxWkEvSFFaR2w3TWRRanhPVHVERGxUOXRwcll1d3ByTTlETlZRZWVubFJaL3ZrbUc2bERQcWFzTTAwQlUvaHEvalI4eWFsblVWZFc0QTdBRzByOGNnazN6WUVsZnh3NUltc1lhek1nemlvb1orNGFOU01ia3p2R3FrTnQ1cERRM2llM1ArYzZDVEp5ZFZnZVpEUkhpYzBPZVU1UWY4dDlBWCs4bnpMTmdZTy9uVm54eXNtN2ZmVyt1czJkSnBNUGxGc1hpUXd5bGVUODkvQUpLUDVrVmIrNmd6TjdXTkJQaE1rZSswcUtIZ2VEWVlUVW8zWTB4MVlQanR6NXNSR1ZzV3A0LzZVWFExeW9GMDNDRVY3Z3o2ZkVQRDB2eTgwM01SUFRWU2dQZGFLUGlkNlk9'
url_north = 'https://lvr.land.moi.gov.tw/SERVICE/QueryPrice/4d1f6903c644a0cbddc0fc23bbe24fb3?q=VTJGc2RHVmtYMS9hdzRrQTdMYW1MRlhwL0d4U25SUnhreERSQ2RDL1pycm1yV1hTOXQ1TzFKZXJDMXFCSkhxLzBFMVJmVFZEbGpicXloSThCcWpVQjdOTGo3OHBsb3BzK0gxYzlPcUFsaG4vb1VpZ0dPTEdVT0VkUHBQaUpkcjg5Ulp2RHRsY0RzVjVnV3BwVmtPUVpUd3FqVm55NnpvMmY2Q1BkQllDYWdFNUM2dk9IN29KUWZmM0s0SWVXOU5zeUtuQnFuY01tVnBLaVVnOGg2OUVSc2lkdjBiNmlrdW9malBZaDFISWxaUTBoSGtNSlZ4N1FIdjlSTzd6Sk5aR3Y5RE0yM2REWk44QmkwemE2K3ZjZERwcVdZS0hzSDIrL056S0xEbFh6eFY1dXk1d1k2VWJHRHdMSnJwUWwrVTlncGZkZDN1a21SOFZqNVJMVGFCRHpaRldiWmY4SzVBK0xOaXNtT1lMTGdvbVdpTjlTZXRPT255VU00MjlocHdyMExQVFpqVE5GZG5Tc01hNDlyK3BiS0NpUkIrbE9zamdjeURFTVREMk5NTnN6VURMTklzMVdkZ2ZtbnRkY1MvaXhEcWI0b0R1TEFwZkZ2TjBIS3lwcnNvWTRLSjliUGd2WnhNdjN0cHp6T1V2bTlGQ2Z4VEF2YzVzN3E5dG9KOUV3VFR0TlhZSVBuL1Q1cDVRN2M3Wm1YTVA4K3dGSGFyTGJhU3BBTERnU2RPR2V3Z0JpMFpQVXM2RjRxQ09QL0M4ZjNJYi9LQk5wTjU2WkJUaXNEU0RDRU9UNk9CSlVNLzU5WDZUN1pHWmpvOXhnRkRWdmE1cnIxTEpRMUt6NXU4Y2pRTWZMSHlrbHRud1FXUHBJdzR5V241OUd1YWRPV2FVeGJXM3MxR2NMRlU9'
url_mid = 'https://lvr.land.moi.gov.tw/SERVICE/QueryPrice/929f1afde498986523b14f9faa921ae4?q=VTJGc2RHVmtYMS9XR3c3N1lSQWRlNzFidDNMWEJsVWVVTEFWamJxUzQ3QjlpM1dBRWN4Y0hsSGpScTIyVU00SHpMK0x4Rnl6bWdCVnNGc2Jvb0JjRG9mc3BPcWRjRlc0TldURUtiQm0yY1pDYithd1NhQW96aGpIMnJJOVJtVWdacHFzbjlEOVBBeEVnekZYa1k0aEF3SWtnOGpTUU1xMFZJQmFRT1c2MWpjb3JBMGc0eXlSeTJlcEFRVlZaY2M1V2JXa3RmNThJWXNZb0s2aDFYOFZVaFpwV0M2aFdudzcyL04xenhWWXY3WXcrOTNuNC9LcWh3SUJWN2g1VGszSGtXbnJVbVdJdWM5YXRkMXlGUE1SWFRvZEZoNkJlZFRGVzNyM2g0RHFQcUlsakFzTVpyM2dRcTF1WEhva1UrU3IwSVQwVUErM0JVSVBPMUtyUzVrL08rUm13cTk1ZlVXNE1jN3Z6Mi9INWd0UWE5LzhKdjNGeVkvWWl3QmVLWCtHMVh6N2taRTdnU2xDWjkxUEpXZEJzTkNyb0RTYWZiM2FFZUlITk53U0ExMm5FcjFTZlVJOW5JZHdQcXUyTjZiaVA0MUNoVTNZVllCY0hRNmJWVjdjWExsbzBEb3B0ajQwNC9TbDlSY0tQK2VFRElvT2wrOG1jSy8rMWc0MFBsa3d0Vy9BaFRSQ0hKaWFJcDVDclRlUGZhaHc1MDhBbXFLM09vVG1Odm5QODBSOVBCRmVJaDR5UG1ab0VQRFkzaGt2VlFiT1J4dm1mOHJmaXR4R1BzS2VkSEp0bXpPTVAxK1F0ZHlsZ3BVamFXRmUrMTRXUm4vdkdYcHBtUXlLZUVBeVI1ZG10UGpFcVFkaUtydTd4bkZBRlY1Qi8rU1ZnNzhOM3MxREJFRkNZVnM9'
url_xituan = 'https://lvr.land.moi.gov.tw/SERVICE/QueryPrice/12d75537988b8e180cbb09fc4e8020ab?q=VTJGc2RHVmtYMThGLzFYY0RmYU9hTDBrRExIUVlJTGx4NGp1M3F5aEVQS3p3K3cxaVdkL1ZDbmd6K3Z2OUV4WFFUOUV6WHAyNjNveHR5QzdyYVArMU90MUo4eWpsSjZQbVp0Y2hBTERuVTJiYTZZSDF2cDhHRm9xdTRnMTBvUTZPR1Zidlg1cFczbVdOU1dtaWV4ZFlxUlRuZk1RNnBBK09yQUFHZ2JOWm9vSm1DVXA0VDl5enlxS0dMZ1ljQjNjSW9sYnl0Y1JSVVV0WlorL2lqV29DYjNubXZVc1Y3RGRrTURoSUJueHl1UktXZThvVjF2MzJjYjV5Y25IODZHVGNrd1VMMXlWbWpKWUNGai9Xa0FqbGR1YnNEYWxwS2ZkY2dpeDBqNzVlMW5KT1IzOERMZ1cvT0ppMGt5STZoSVQ5KytrT1hIUWpnVm1hcHdIMWNXUGI2YmtHbzVuSjJhM3h6MmNpQlZSSHF3Z0VRUWZRU3ZBMTdKbnhvVTZOQWs0UGNldjBuWXN2SUlHb1pUVElrRjFmajZNTTE5bnZXL3pKSXVZSzFFTlRmajZjckljVTNNRFJWWmFTdllqZHFCT3M3MFR1SzhXNlErZEJQbnNIYWJ3VlpBZkFtSVVvWkdrTWdRZk90VDc4ZTUwSkJ0cnBVazgyOTJpK2VyMFJVeGNlRk52ZThNYTFSTDJBeWU2UVM1eS9IOUl5TEoyNERoVnd5d0c0RlBKdW1Wcjc3UTR6TElHUGtFSDVkUEM1emVNS3VGZkxYQnFDckJuSm03eG1JKzd4L3ZweE90VzF3UVhQVzZNcnlCU0ZQK1pWcE1mN29BcTNLQ3UvWlpUdDVVSDhMaUZwN3hWbnMzeG9xZHUwRGIzRnhlZkJvdDloOVROVU9mbzJiaVoxMzQ9'
url_beitun = 'https://lvr.land.moi.gov.tw/SERVICE/QueryPrice/67196315597824c33d2c0d5eabdd3279?q=VTJGc2RHVmtYMStVQ0MzN2Nzb0p1MmtReHVESTlmMkhXb2dQTzFFSFhRR1g4dlNLb1lmVy9MMys4OEV3cy80YVk0aVhucENvZTFZZG5wd1d3U1pHeW9XemE5YldERUQzN0lOZjkzL3hUUHpkVnlTWnhQNUk2a1RhSURIcGhRZW5OY3pMVkJ6dlBBR0gzSG5hSitXeUgvVmZMY2lpbXhXR1hkcDh6dU9Jbk56TkRqVWtuOFZUMkJSR3E5RytmalhxZFI1SCtzTjVUbTd4RE0rK2tlby83UHdaYTJNWTdjZzZEcU5jWmUxNU45R1J2aC93VFJlQmt4L1hBSy81SnlMb3JLQU0xTGlmMTM0SmlSMlA2d0ZPeitRME5kWmovaHUxZ3hVbFZaMnJBd0l2UUJZUTUrVFk5USt1YTBQY1lTOFdpMHUzWkd0OTBldExjSkVPbTFRb1A1cEJyUG1MekFFNy90d1p4NGpNN0hKaHN1OW5tWDFRb2thM2pXN1puTDVsYmNQTE52dWlYTnhTdXRxUk1HSVJoUVFHbUtPbDJuQWlyOSs5RTNDdXhvR0Z2aUtOZ2VlTUR1SldHOEZJUmhyRWF4UzhLNDgxWEhqeUpZdVBLWnBHUVVTR2p6eURSS2RZWnhYaDVqc3h3Y0ZBNktEOElpekdLNnRJWk56cE9lanRLd2h0K2ZidDJMWGd4NndtbGJvcUFqcFZkWUJUbDZYaXNuWExjRGR1R3h6c28rbHlDMjhmSWR0U2t0Uk9hOXFIVXZDdjBLeVRGUmlGbTdxL3RuZHBYdjQ2L2x6ZURBVjBuTXJwZ3NIQmJhaHhMRDZFQzF5THB5ZE41VjU4a09rdTZmTFdtN0JybVNGNXNkaVJjbXc4elNVQzROOGthY3lCTlNGSG9ZQ2FsWXc9'
url_nantun = 'https://lvr.land.moi.gov.tw/SERVICE/QueryPrice/acaf9e23229f58ca0b70d694d0390cba?q=VTJGc2RHVmtYMThVRHlFU0NVaTVTVnoxZjFhYWt4V3lVU0MwMkdkUXdFaGZlOStBcUM2K0RSS2MvUEV6MFduNWYrUFJtRGN6VWRxcTJmSWJ2RlhNZmhneWx4ZEFWTVpENG1takhmUmRrZmdzWE40N0YycUMxZkZZSS9jdHlGVE5ObjdVZ2R1K0xncG5WN3FIZWp4Y1FnNGxRYXBzK1dzbExEci9pWUFlNHZCay9EOUEzMmtuN2p5eDIrV0h4MWNjMUQ2YzY4Q2I4OHc5U2ZJNmZOTklUOFNVRis0UnU3dTZiTzdmTHFyajM4Ry82SGw1ZktSUkhDYVJrblZleEtZMEZ1ai9MSG1zR3lVbWNKazhOUkhZanc5djdKaXhOSGovUGNTclV6WTlJZjlDU0dEdzdqNy9oTWU4QUNZYlpxYVdBUUZ5VHp5QmtFOXZCVWYyMXRYd3poMXNXVkNnNE5STWhZYUZSdE9JSnJqdGdmVzVLVlRiYW0vS1N2OG5GV1lMZlpyUEZVVFRrNk5FRnVlMGFPM0xYSlVOOFNyaDZaSE1iMFBlWllWTVBsVEdTaENZL0dtemxpU3daNVYwNFh2M1d2c0VZOUM5dEprVDAwT1YrMDVDaWVqZ09aN2dDb1E3c0VnUGNNVFBySXl0NzBlaFd6Y1J5UTB6dHh4bmlYVmhDa3htdk5XYjN3d1R5a1NTaGd0NnBHeEs4K3NBbTA1ZGVUL1RTV3JZU2FvemNEcFVDQkVTQzgzU3hmTVNYRU10WG83bHd1dnlhK0NHd2Y5NGVoK1F2R3lSckNKOFQvZzlXZU83RkJ2V3VzRTJXQ2s2TG9vMU9kT2F2MG1mdUdYb2ZuWlJRem1LckVMTkdudG1jUTMzRjBxb2k0am1nbHorNHczbnF6YXdyWW89'
url_list = [url_east, url_west, url_south, url_north, url_mid, url_xituan, url_beitun, url_nantun]

dataframe_list = []
for url, district_name in zip(url_list, district_name_list):
    dataframe = web_crawling_lvr_land(url=url, headers=header)
    dataframe = create_district_name(dataframe, district_name)
    dataframe_list.append(dataframe)
    time.sleep(15)
    
df = pd.concat(dataframe_list)
df_filtered = df[['a', 'b', 'bn', 'bs', 'e', 'es', 'el', 'f', 'g', 'lon', 'lat', 'pu', 'm', 'note', 'tp', 's', 'p','punit', 'v', 'district']]
df_filtered.rename({'a':'address', 'b':'build_type', 'bn':'community', 'bs':'build_share1', 'e':'deal_date', 'es':'build_share2', 'el':'elevator', 'f':'floor', 'g':'age', 'lon':'longitude', 'lat':'latitude', 'pu':'main_purpose', 'm':'manager', 'tp':'price', 's':'plain', 'p':'unit_price','punit':'parking_unit', 'v':'layout'}, axis=1, inplace=True)
df_filtered.to_csv('C:\\Users\\Jacob Kai\\Documents\\Python_UniClass\\hw1_2023\\Taichung_RealEstates')
