import axios from 'axios';

// .env 파일의 API 주소를 불러옵니다.
const apiClient = axios.create({
    baseURL: process.env.REACT_APP_API_URL, // .env에 저장된 주소
    headers: {
      'Accept': 'application/json',
    },
  });
  

export const getOkrData = (page, company_name, field, new_sorting) => {
    const params = {
        ...(company_name && company_name.trim() !== '' && { company_name }), // 빈 문자열 제외
        ...(field && field.trim() !== '' && { field }),                     // 빈 문자열 제외
        ...(new_sorting && { new_sorting }),
    };

    return apiClient.get(`/${page}`, { params });
}
export const getTotalAIData = (page, company_name, field, new_sorting, page_size) => {
    return apiClient.get(`/prediction/${page}`, {
        params: {
            ...(company_name && company_name.trim() !== '' && { company_name }), // 빈 문자열 제외
            ...(field && field.trim() !== '' && { field }),                     // 빈 문자열 제외
            ...(new_sorting && { new_sorting }),
            ...(page_size &&  page_size !== 0 &&{ page_size })
        },
    });
};
export const getUniqueAIData = (okr_id) => {
    return apiClient.get(`/ai/${okr_id}`);
}
export const postAIData = (okrIds) => {
    return apiClient.post('/ai/', {"okr_ids": okrIds});
}
export const postExcel = async (file) => {
    const formData = new FormData();
    formData.append('file', file); // 서버가 요구하는 필드명 "file"로 파일 추가
  
    try {
      // apiClient를 사용하여 POST 요청
      const response = await apiClient.post('/upload/', formData, {
        headers: {
          'Content-Type': 'multipart/form-data', // multipart/form-data 설정
        },
      });
      console.log('파일 업로드 성공:', response.data);
      return response.data;
    } catch (error) {
      if (error.response) {
        console.error('서버 응답 에러:', error.response.data); // 서버에서 반환한 에러 메시지
        throw new Error(`업로드 실패: ${error.response.data?.message || '알 수 없는 에러'}`);
      } else {
        console.error('요청 중 에러 발생:', error.message);
        throw new Error('파일 업로드 중 문제가 발생했습니다.');
      }
    }
  };
  
export const getCompanyData = (page, company_name, field, page_size) => {
    const params = {
        ...(company_name && company_name.trim() !== '' && { company_name }), // 빈 문자열 제외
        ...(field && field.trim() !== '' && { field }),                     // 빈 문자열 제외
        ...(page_size &&  page_size !== 0 &&{ page_size }) 
    };

    return apiClient.get(`/company/${page}`, { params });
}

export const getTaskStatus = (task_id) => {
    return apiClient.get(`/tasks/${task_id}`);
}

export default apiClient;