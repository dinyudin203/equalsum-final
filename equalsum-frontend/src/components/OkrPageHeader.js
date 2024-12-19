import React, { useState } from 'react';
import '../styles/OkrPageHeader.css';
import ExcelImportPopup from './ExcelImport';

const OkrPageHeader = ({ activeTab, setActiveTab, onExcelImport }) => {
  const [isPopupVisible, setIsPopupVisible] = useState(false);

  const handleOpenPopup = () => {
    setIsPopupVisible(true);
  };

  const handleClosePopup = () => {
    setIsPopupVisible(false);
  };

  return (
    <header className="header-container">
      {/* 첫 번째 줄: 제목 */}
      <h1 className="header-title">Okr AI consultant</h1>

      {/* 두 번째 줄: 네비게이션 + Import 버튼 */}
      <div className="nav-section">
        {/* 왼쪽: 네비게이션 버튼 */}
        <div className="nav-buttons">
          <button
            onClick={() => setActiveTab('OkrInfoPage')}
            className={activeTab === 'OkrInfoPage' ? 'active' : ''}
          >
            OKR 기업정보
          </button>
          <button
            onClick={() => setActiveTab('OkrDataPage')}
            className={activeTab === 'OkrDataPage' ? 'active' : ''}
          >
            OKR 데이터 목록
          </button>
          <button
            onClick={() => setActiveTab('OkrAIPage')}
            className={activeTab === 'OkrAIPage' ? 'active' : ''}
          >
            AI 적용
          </button>
          <button
            onClick={() => setActiveTab('OkrAITotalPage')}
            className={activeTab === 'OkrAITotalPage' ? 'active' : ''}
          >
            저장된 AI 결과
          </button>
        </div>

        {/* 오른쪽: Import 버튼 (조건부 렌더링) */}
        {activeTab === 'OkrInfoPage' && (
          <button className="import-button" onClick={handleOpenPopup}>
            Import
          </button>
        )}

      </div>

      {/* 팝업 컴포넌트 */}
      <ExcelImportPopup
        isVisible={isPopupVisible}
        onClose={handleClosePopup}
        onExcelImport={onExcelImport}
      />
    </header>
  );
};

export default OkrPageHeader;
