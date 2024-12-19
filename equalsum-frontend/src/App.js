import React, { useEffect, useState } from 'react';
import OkrPageHeader from './components/OkrPageHeader';
import OkrInfoPage from './components/OkrInfoPage';
import OkrDataPage from './components/OkrDataPage';
import OkrAIPage from './components/OkrAIPage';
import OkrPageFooter from './components/OkrPageFooter';
import OkrAITotalPage from './components/OkrAITotalPage';
import './styles/global.css';

const App = () => {
  const [activeTab, setActiveTab] = useState('OkrInfoPage'); // Default: OKR 기업정보 페이지
  const [aiOkrId, setAIOkrId] = useState(0); // 태스크 아이디 관리
  const [aiTaskStatus, setAITaskStatus] = useState(''); // 태스크 상태 관리

  /*useEffect(() => {
    if (aiTaskStatus === 'success' && activeTab === 'OkrAIPage') {
        setActiveTab('OkrAITotalPage');
    }
  }, [aiTaskStatus]);*/

  return (
    <div>
      {/* Header Component */}
      <OkrPageHeader activeTab={activeTab} setActiveTab={setActiveTab} />

      {/* Main Content */}
      <main style={{ padding: '20px' }}>
        {activeTab === 'OkrInfoPage' && (<OkrInfoPage/>)}
        {activeTab === 'OkrDataPage' && (
          <OkrDataPage
            setActiveTab={setActiveTab} // Data -> AI 페이지 전환을 위해 추가
            setAITaskStatus={setAITaskStatus}
            setAIOkrId={setAIOkrId}
          />
          )
        } 
        {activeTab === 'OkrAIPage' && <OkrAIPage aiOkrId={aiOkrId}/>}
        {activeTab === 'OkrAITotalPage' && <OkrAITotalPage aiTaskStatus = {aiTaskStatus}/>}
      </main>

      {/* Footer Component
      <OkrPageFooter /> */}
    </div>
  );
};

export default App;
