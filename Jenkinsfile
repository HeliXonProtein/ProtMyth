pipeline {
    agent any

    environment {
        CONDA_BASE_PREFIX = "$HOME/miniconda3"
        ENV_PREFIX = "$WORKSPACE/env"
        PATH = "$ENV_PREFIX/bin:$CONDA_BASE_PREFIX/bin:$PATH"
    }

    stages {
        stage('Build') {
            steps {
                echo 'Begin build stage...'
                sh '.ci/autosetup.sh'
            }
        }
        stage('Static Test') {
            steps {
                sh 'rm -rf reports && mkdir -p reports'
                echo 'Flake8 static check...'
                catchError(buildResult: 'UNSTABLE', stageResult: 'UNSTABLE') {
                    sh 'timeout -s SIGKILL 600s flake8 --format=pylint src tests > reports/flake8.log'
                }
                echo 'Mypy static check...'
                catchError(buildResult: 'UNSTABLE', stageResult: 'UNSTABLE') {
                    sh 'timeout -s SIGKILL 600s mypy src tests > reports/mypy.log'
                }
                echo 'Header check...'
                catchError(buildResult: 'UNSTABLE', stageResult: 'UNSTABLE') {
                    sh 'python .ci/copyright_header_checker.py src tests -o reports/copyright-check.log'
                }
            }
            post {
                always {
                    recordIssues(
                        qualityGates: [[threshold: 1, type: 'TOTAL', unstable: true]],
                        tools: [flake8(pattern: 'reports/flake8.log')]
                    )
                    recordIssues(
                        qualityGates: [[threshold: 1, type: 'TOTAL', unstable: true]],
                        tools: [myPy(pattern: 'reports/mypy.log')]
                    )
                    recordIssues(
                        qualityGates: [[threshold: 1, type: 'TOTAL', unstable: true]],
                        tools: [flake8(id: 'copyright-header', name: 'Copyright Header', pattern: 'reports/copyright-check.log')]
                    )
                }
            }
        }
        stage('Spelling and Doc Test') {
            steps {
                echo 'Pylint code style and documentation check...'
                catchError(buildResult: 'UNSTABLE', stageResult: 'UNSTABLE') {
                    sh 'timeout -s SIGKILL 600s pylint --rcfile=.pylintrc src tests > reports/pylint.log'
                }
            }
            post {
                always {
                    junit(
                        testResults: 'reports/pylint.log'
                    )
                }
            }
        }
        stage('Unit Test') {
            steps {
                echo 'Unittesting...'
                sh 'pytest --junit-xml=reports/unittest.xml ./tests'
            }
            post {
                always {
                    junit(
                        testResults: 'reports/unittest.xml'
                    )
                }
            }
        }
        stage('Deploy') {
            steps {
                echo 'No deploy stage needed...'
            }
        }
    }
}