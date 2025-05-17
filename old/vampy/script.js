const modules = [
  { id: 0, title: "Preparation", time: "Before 9:30", completed: true },
  { id: 1, title: "Introduction", time: "9:30 - 10:00", completed: true },
  { id: 2, title: "What is Computer Programming", time: "10:00 - 10:30", completed: true },
  { id: 3, title: "Navigating this Course", time: "10:30 - 11:00", completed: false },
  { id: 4, title: "Running Your First Program", time: "11:00 - 11:30", completed: false },
  { id: 5, title: "How to Succeed in Coding", time: "11:30 - 12:00", completed: false },
  { id: 6, title: "Data in Python", time: "12:00 - 12:30", completed: false },
  { id: 7, title: "Combining Text and Calculations", time: "12:30 - 13:00", completed: false },
  { id: 8, title: "Variables", time: "13:00 - 13:30", completed: false },
  { id: 9, title: "Building LLM Prompts with Variables", time: "13:30 - 14:00", completed: false },
  { id: 10, title: "Functions: Actions on Data", time: "14:00 - 14:30", completed: false },
  { id: 11, title: "Build a Web Interface", time: "14:30 - 15:00", completed: false },
  { id: 12, title: "Demo Your App", time: "15:00 - 15:30", completed: false },
  { id: 13, title: "Get your Certificate", time: "After completion", completed: false },
]

document.addEventListener("DOMContentLoaded", () => {
  const loginForm = document.getElementById("login-form")
  const loginScreen = document.getElementById("login-screen")
  const courseScreen = document.getElementById("course-screen")
  const logoutBtn = document.getElementById("logout-btn")
  const modulesList = document.getElementById("modules-list")

  loginForm.addEventListener("submit", (e) => {
    e.preventDefault()
    loginScreen.classList.add("hidden")
    courseScreen.classList.remove("hidden")
    renderModules()
  })

  logoutBtn.addEventListener("click", () => {
    courseScreen.classList.add("hidden")
    loginScreen.classList.remove("hidden")
  })

  function renderModules() {
    modulesList.innerHTML = ""
    const nextModuleIndex = modules.findIndex((module) => !module.completed)

    modules.forEach((module, index) => {
      const isDisabled = index > 0 && !modules[index - 1].completed
      const isNext = index === nextModuleIndex

      const moduleCard = document.createElement("div")
      moduleCard.className = `module-card ${isDisabled ? "disabled" : ""} ${isNext ? "next" : ""}`

      moduleCard.innerHTML = `
                <div class="module-header">
                    <div class="module-title">
                        <span class="module-number">${index.toString().padStart(2, "0")}</span>
                        <div class="module-info">
                            <h3>${module.title}</h3>
                            <div class="module-time">
                                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><polyline points="12 6 12 12 16 14"></polyline></svg>
                                <span>${module.time}</span>
                            </div>
                        </div>
                    </div>
                    <div class="module-status">
                        ${
                          module.completed
                            ? '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg>'
                            : isDisabled
                              ? '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="11" width="18" height="11" rx="2" ry="2"></rect><path d="M7 11V7a5 5 0 0 1 10 0v4"></path></svg>'
                              : ""
                        }
                    </div>
                </div>
            `

      if (!isDisabled) {
        moduleCard.addEventListener("click", () => {
          alert(`Navigating to Module: ${module.title}`)
        })
      }

      modulesList.appendChild(moduleCard)
    })
  }
})

