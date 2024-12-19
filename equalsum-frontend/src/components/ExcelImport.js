import React, { useState } from 'react';
import { postExcel } from '../api/api.js'; // 서버 요청 함수
import '../styles/ExcelImport.css';

const ExcelImportPopup = ({ isVisible, onClose }) => {
  const [selectedFileName, setSelectedFileName] = useState(null); // 선택된 파일 이름
  const [responseMessage, setResponseMessage] = useState(null); // 성공 메시지
  const [responseError, setResponseError] = useState(null); // 에러 메시지

  // 엑셀 파일 업로드 처리
  const handleExcelUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) {
      console.error('파일이 선택되지 않았습니다.');
      setResponseError('파일이 선택되지 않았습니다.');
      return;
    }

    // 파일 이름 업데이트
    setSelectedFileName(file.name);

    try {
      const response = await postExcel(file); // 서버 요청 함수
      if (response.message === 'success') {
        setResponseMessage('파일 업로드가 성공적으로 완료되었습니다.');
        setResponseError(null); // 에러 초기화
      } else if (response.error) {
        setResponseError(`파일 업로드 실패: ${response.error}`); // 서버 에러 메시지 출력
        setResponseMessage(null); // 성공 메시지 초기화
      }
    } catch (error) {
      // 네트워크 문제 등 서버 자체 에러 처리
      setResponseError(`파일 업로드에 실패했습니다. 서버 응답: ${error.message || '알 수 없는 오류'}`);
      setResponseMessage(null);
    }
  };

  // 팝업 닫힐 때 상태 초기화
  const handleClose = () => {
    setSelectedFileName(null);
    setResponseMessage(null);
    setResponseError(null);
    onClose(); // 부모 컴포넌트의 onClose 함수 호출
  };

  // 팝업이 보이지 않으면 null 반환
  if (!isVisible) return null;

  return (
    <div className="overlay">
      <div className="popup">
        {/* 제목과 파일 선택 버튼 */}
        <div className="header-container1">
          <h3 className="header1">엑셀 파일 업로드</h3>
          <input
            type="file"
            accept=".xlsx, .xls"
            onChange={handleExcelUpload}
            id="file-upload"
            className="file-input"
          />
          <label htmlFor="file-upload" className="file-button">
            파일 선택
          </label>
        </div>

        {/* 선택된 파일명 표시 */}
        {selectedFileName && <p className="file-name">선택된 파일: {selectedFileName}</p>}

        {/* 성공 메시지 표시 */}
        {responseMessage && (
          <div style={{ color: 'green', marginTop: '10px' }}>
            {responseMessage}
          </div>
        )}

        {/* 에러 메시지 표시 */}
        {responseError && (
          <div style={{ color: 'red', marginTop: '10px' }}>
            {responseError}
          </div>
        )}

        {/* 닫기 버튼 */}
        <div className="actions">
          <button className="close-button" onClick={handleClose}>
            닫기
          </button>
        </div>
      </div>
    </div>
  );
};

export default ExcelImportPopup;
