import requests
login_data = {
    'Username':'B11007050',
    'Password':'zxc507ASDFG...'
}
session = requests.session()
login_response = session.post('https://stuinfosys.ntust.edu.tw/NTUSTSSOServ/SSO/Login/CourseSelection',data=login_data)
print(login_response.status_code)